#!/usr/bin/env python3
"""
NMR-guided structure resolution demo — gradient-based, fitted against SPECTRA.

Same experiment as the ``full_tensors`` variant, but the loss is the mismatch between
the predicted and target *spectra* rather than between per-atom tensors. That matters
because a spectrum is the only thing a real experiment gives you: the per-atom tensors
and their orientation are never observed.

The price is a drastic loss of information. The tensor loss compares 9 numbers per
atom; a fast-MAS (or solution) spectrum compresses each site to its isotropic value
``σ_iso = tr(σ)/3`` and then sums the sites into a single curve. This script exists to
find out whether that is still enough to locate one atom in the easy, controlled QM9
case, before the method is trusted on a real solid.

Two consequences are built into the defaults:

- **Every nucleus is fitted separately.** ¹³C and ¹H are different experiments at
  different frequencies; their spectra are never added together. `SPECTRUM_SPECIES`
  lists which ones enter, and the loss is their mean. Adding species is the cheapest
  way to buy back information, since a nucleus far from atom k still responds to it.
- **The default metric is Wasserstein, not RMSE.** A point-by-point spectral loss goes
  completely flat once the predicted peak stops overlapping the target one — for a
  narrow line that happens after a fraction of an Ångström, and the optimiser is left
  with no gradient at all. The transport distance keeps growing with separation, so
  the basin of attraction is the whole axis. See `e3response.nmr_spectra`.

Steps:
  1. Ground-truth spectra (from DFT labels or from the model at the true geometry).
  2. Gaussian perturbation of atom k with validity checks.
  3. Loss = spectral mismatch, averaged over the fitted nuclei.
  4. Gradient-based minimisation: N parallel Adam descents (vmap) over atom k's 3 DOF.
  5. Evaluation: |Δr|, loss curve, overlaid spectra; per-run GIF.
  6. (Optional) Statistics over many molecules.
"""

import csv
import datetime
import functools
import logging
import os
import sys
import typing

# Silence XLA/TSL C++ logging (e.g. the GPU autotuning "dot_search_space" warnings).
# Must be set BEFORE jax/XLA are imported to take effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hydra
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import omegaconf
import reax
import scipy.optimize
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.geometry import jax_neighbours

from e3response import keys, nmr_spectra, structure_search as ss
from e3response.data import qm9_nmr

logging.getLogger("reax").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Model checkpoint directory (must contain config.yaml and checkpoints/last.ckpt).
RUN_DIR = "/home/mattia/Desktop/ml_codes/nmr_project/nmr_diff_runs/qm9_full_heavy/nequip_nmr"

# Dataset — DATASET/R_MAX/train_val_test_split come from RUN_DIR/config.yaml (cfg.data
# below), so the reproduced split matches the run. Only data_dir is overridden, since the
# run's config.yaml may store a relative/cluster path.
DATA_DIR = "/home/mattia/Desktop/ml_codes/CAMML/e3response/data/qm9_nmr"
SPLIT    = "test"      # which partition to work on: "train" | "val" | "test" | "full"
LIMIT    = None       # base dataset restriction, applied BEFORE splitting (None → use the
                      # run's own `limit`, required to reproduce ITS split exactly). Same
                      # syntax as before: int N, "n:m", or "n:m:step".
SPLIT_LIMIT = 110    # further restriction WITHIN the chosen SPLIT (ignored if SPLIT="full"),
                      # e.g. SPLIT_LIMIT=20 → only the first 20 test structures,
                      # "10:30" → test structures 10-29. Only extracts/parses those.

# Demo: which molecule and atom
MOL_IDX = 1      # index into the chosen limit or SPLIT and limit (not the raw dataset)
ATOM_K  = "C"    # None → random atom; int → direct index; str → random atom of that species (e.g. "N")
SEED    = 42     # fixed seed for reproducibility

# Perturbation (BENCHMARK ONLY — a real experiment has no "true" position to perturb
# away from; there the search is seeded from a candidate site in a structural model).
SIGMA                = 0.8   # Gaussian noise amplitude (Å)
MIN_DIST_PERTURB     = 1.0   # min distance to any other atom for a start to be ACCEPTED (Å).
                             # No two QM9 atoms are closer than ~1.09 Å (C–H), so anything
                             # below that is off the training manifold and the benchmark is
                             # asking the model a question it was never taught. The old
                             # shared value of 0.8 Å admitted starts at 0.797–0.855 Å.
MAX_PERTURB_ATTEMPTS = 200   # retries for valid perturbation

# Validity guard on the loss itself — this one DOES survive to deployment, since the
# optimiser can wander into a clash whatever seeded it. Distinct from MIN_DIST_PERTURB:
# its only job is to fence off the region where the model stops being a function.
# Measured on this checkpoint by pushing an atom onto its nearest neighbour:
#   1.50 Å → loss   6.7, |grad| 8.5e+01, max|σ| 2.3e+02 ppm
#   0.80 Å → loss   5.8, |grad| 4.6e+01, max|σ| 2.3e+02 ppm
#   0.52 Å → loss  19.7, |grad| 3.1e+02, max|σ| 2.3e+02 ppm      ← still well-behaved
#   0.45 Å → loss  76.9, |grad| NaN                              ← blow-up starts here
#   0.40 Å → loss 146.6, |grad| NaN,     max|σ| 2.1e+16 ppm
#   0.20 Å → loss 108.1, |grad| 5.2e+12, max|σ| 2.8e+20 ppm
# The previous 0.8 Å was far too conservative: on the straight line from start to truth it
# declared 48/60 samples invalid for mol 2, 24/60 for mol 6, 18/60 for mol 10 — walling off
# space the model evaluates perfectly well and flooding the run with `inf`.
MIN_DIST_GUARD = 0.5
# Rather than a bare `inf` wall, approaching the guard costs a smooth quadratic penalty
# starting GUARD_MARGIN above it, so the gradient PUSHES the atom out of a clash instead of
# leaving the optimiser with a zero gradient on an infinite plateau. `inf` is still returned
# for genuinely undefined geometries (non-finite position, atom with no neighbours at all),
# which no gradient can repair.
GUARD_MARGIN = 0.15   # Å above MIN_DIST_GUARD at which the penalty switches on
GUARD_WEIGHT = 200.0  # penalty = GUARD_WEIGHT · (MIN_DIST_GUARD + GUARD_MARGIN − d)²

# Loss function
# "dft"   → target = spectrum built from the DFT labels in the graph (node["nmr_tensors"])
# "model" → target = spectrum of the model's OWN prediction at the ORIGINAL geometry
#            (tests self-consistency of the landscape, decoupled from DFT accuracy)
LOSS_TARGET = "dft"

# ── Spectrum ──────────────────────────────────────────────────────────────────
# Which nuclei are fitted. Each is an INDEPENDENT experiment: one spectrum and one
# loss term per species, averaged — never summed into a single curve. ("C",) mimics a
# ¹³C measurement; ("C", "H") adds a ¹H one, which constrains atom k much more because
# every hydrogen in the molecule responds to it. Species absent from a molecule are
# skipped with a warning.
SPECTRUM_SPECIES = ("C", "N")
FWHM        = 2.0          # ppm — linewidth. Real ¹³C solution lines are far narrower;
                           # this stands in for the experimental + model uncertainty.
LINESHAPE   = "gaussian"   # "gaussian" | "lorentzian" | "pseudo_voigt"
# Metric. "wasserstein" (ppm, transport distance) is the default because the
# point-by-point metrics are FLAT once peaks stop overlapping — see the module
# docstring. "rmse"/"mse"/"cosine" are kept for comparison.
METRIC      = "wasserstein"
NORMALISE   = "area"       # "area" | "max" | "none" — experimental intensities are
                           # arbitrary units, so the raw amplitudes are meaningless.
# Referencing δ = slope·σ + intercept. The default merely mirrors the axis (δ = −σ),
# which is self-consistent here since target and prediction use the same convention.
# Fitting REAL data requires calibrating these against known compounds.
REF_SLOPE     = -1.0
REF_INTERCEPT = 0.0
# The grid must stay wide enough to hold the peaks for the WHOLE optimiser excursion:
# if a line leaves the window, its area vanishes, normalisation blows up and both the
# loss and its gradient become artefacts rather than signal.
GRID_MARGIN = 60.0         # ppm of padding beyond the target's own peak range
GRID_POINTS = 1024

# ── Optimisation ──────────────────────────────────────────────────────────────
# Adam, on device, over the 3 DOF of atom k. The loss is differentiable end-to-end (the
# model recomputes edge vectors from positions), so the gradient chain pos_k → tensors →
# spectra → loss is exact.
#
# This USED to be `gcnn.adapt` + `jax.scipy.optimize.minimize(method="BFGS")`. That was
# measured to be the single dominant cause of the 1/10 recovery rate: benchmarked on 10
# QM9 molecules it returned `status=3` — which is `2 + line_search_status 1`, i.e. *zoom
# failed* — in 10 cases out of 10, after a median of 12 function evaluations, with |grad|
# at the returned point ranging from 1.2 to 449 against a `gtol` of 1e-5. It never
# converged once; it always died in the strong-Wolfe line search. The reason is precision:
# the float32 model gives the loss ~4e-3 relative jitter, and the curvature condition
# |g·p| ≤ 0.9·|g₀·p| simply cannot be evaluated reliably at that noise level.
#
# Adam has no line search and takes gradient-magnitude-normalised steps, so the jitter
# cannot stall it. On identical starting points it beat BFGS on the final loss in 10/10
# cases while running 4.6× faster. Four other suspects were tested and refuted along the
# way — FWHM annealing, the guard threshold, the displacement cap, the live neighbour
# list — none of them changed the outcome on its own.
METHOD_LABEL = "adam+multistart"
ADAM_STEPS   = 200     # iterations per start
ADAM_LR0     = 0.08    # Å-scale step at the beginning (cosine-decayed to ADAM_LR1)
ADAM_LR1     = 0.004   # final step: sets the resolution the answer is polished to
PRINT_EVALS  = False   # per-start progress. With a vmapped multistart this is a firehose;
                       # the per-start summary printed at the end is usually what you want.

# Multistart. The landscape has many local minima and BFGS's replacement does not change
# that: measured over the 10-molecule benchmark, only 13 of 320 random starts (4%) land in
# the basin of the true position. The old scheme — jitter the ONE perturbed guess by 0.4 Å
# and retry — never left the wrong basin, which is why 5 restarts helped exactly as little
# as 1. Starts are now drawn over a whole ball and, since the graph has a fixed shape, run
# in PARALLEL through `jax.vmap`: 32 starts cost ~3.5× three sequential ones, i.e. ~10×
# less per start.
#
# N_STARTS is a probability budget, not a performance knob. With a ~1/32 per-start hit
# rate, P(at least one hit) = 1 − (1 − 1/32)^N:  32 → 64%,  64 → 87%,  96 → 95%.
# Anything below ~32 is a coin flip; that is the honest reading of the measurement.
N_STARTS     = 64
# How many of them are in flight at once. `vmap` replicates the model's activations per
# lane, and reverse-mode AD holds them for the backward pass, so memory grows with
# lanes × atoms: 64 lanes on a 23-atom molecule asked XLA for 5.1 GB and died. Batching
# costs nothing — every full batch shares one compilation — and bounds the footprint
# independently of N_STARTS. Lower it if a bigger structure still runs out of memory.
STARTS_PER_BATCH = 16
START_RADIUS = 2.5     # Å — starts are drawn uniformly in a ball of this radius around the
                       # seed. Must cover where the atom might actually be: here ~3σ of the
                       # perturbation, in deployment the uncertainty on the candidate site.
# Each start then searches only its own neighbourhood: u is mapped to
# pos_k = start + R·tanh((u − start)/R), so |pos_k − start| < R for any u. Coverage is the
# multistart's job, not this cap's. Note the Jacobian is sech²(r/R) = 1 − (|Δr|/R)², so a
# target sitting at 0.97 R reaches the optimiser with a 17× damped gradient — keep R
# comfortably larger than the distance any single start is expected to travel.
LOCAL_CAP = 1.5

# Degeneracy detection — the one deployment-relevant output of the multistart. Against a
# real spectrum there is no true position to compare against, so a confidently wrong answer
# is indistinguishable from a correct one: on this benchmark two molecules converged to a
# loss equal to (mol 3: 1.84 vs 1.79) or better than (mol 6: 0.90 vs 2.41) the loss at the
# TRUE geometry while sitting 0.87 and 0.79 Å away from it. What IS observable is that the
# multistart found several well-separated positions with near-equal loss. When it does, the
# site is not determined by the spectrum and must be reported as unresolved rather than
# answered.
DEGENERACY_SEP = 0.5   # Å — how far apart two minima must be to count as distinct
DEGENERACY_TOL = 0.25  # relative loss gap below which the alternative is "just as good"

# Neighbour list. The optimiser's graph is built once per run and only `nodes.positions`
# changes inside the jitted loop, so a list frozen at the start goes stale as atom k moves:
# pairs that should enter the cutoff are simply absent and the loss drifts away from the
# true one (measured on CSH: 4.33 vs 0.60 after 0.557 Å). `tensorial.geometry.jax_neighbours`
# solves this properly — `jnp.argwhere(..., size=capacity)` searches neighbours with a FIXED
# output shape, so the list can be recomputed at every evaluation inside jit, with no
# recompilation. Verified against `graph_from_ase`: identical edge sets (256/256 on QM9,
# 2790/2790 on CSH), agreement to 0.08 ppm, gradient FD-checked, ~1.3 ms per call.
#
# Unlike a padded "skin", this needs no bound on how far the atom may travel and stays exact
# when EVERY atom moves — which is what resolving several Na positions at once will require.
LIVE_NEIGHBOURS = True
# Slots per atom for the fixed-shape list. None → estimated from the density, clamped to the
# atom count, then raised if the starting geometry already overflows. Too few slots silently
# DROPS neighbours, so `did_overflow` is checked on the host before each run.
NEIGHBOUR_CAPACITY = None

# Loss ceiling for a recovery to count as "reliable" in STEP 6 — the observable-only
# proxy for confidence, since a real run has no ground truth to check against.
# UNITS FOLLOW `METRIC`: ppm of transport for "wasserstein", dimensionless for
# "rmse"/"cosine" — retune when you switch. A good anchor is the loss reached at the true
# geometry, which STEP 3 prints; on this benchmark that is 0.39–2.41 ppm depending on the
# molecule, so a single global threshold is necessarily crude.
RELIABLE_LOSS_TOL = 2.0

# Step 6 – statistics loop
RUN_STATS      = True  # set False to skip the multi-molecule statistics loop
N_MOL_STAT     = 20     # how many molecules to include (ignored if RUN_STATS=False)
N_ATOM_STAT    = 1      # atoms per molecule
DIST_THRESHOLD = 0.2    # Å – "success" criterion

# Output
SAVE_CSV        = True   # write trajectory + summary CSV files
SAVE_GIF        = True   # write one GIF per model (can be slow for many steps)
SAVE_SPECTRA_CSV = True  # write the target / perturbed / recovered spectra as columns
GIF_MAX_FRAMES  = 120    # subsample trajectory to at most this many animation frames
GIF_FPS         = 15     # frames per second

# Each run writes to ./results/<timestamp>/ so nothing is overwritten.
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
RUN_ID      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(OUT_DIR, "results", RUN_ID)
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOT_PATH            = os.path.join(RESULTS_DIR, "structure_resolution_result.png")
CSV_TRAJ_PATH        = os.path.join(RESULTS_DIR, "structure_resolution_trajectory.csv")
CSV_SUMMARY_PATH     = os.path.join(RESULTS_DIR, "structure_resolution_summary.csv")
GIF_PATH             = os.path.join(RESULTS_DIR, "structure_resolution.gif")
CSV_SPECTRA_PATH     = os.path.join(RESULTS_DIR, "spectra.csv")
STAT_PLOT_PATH       = os.path.join(RESULTS_DIR, "structure_resolution_statistics.png")
STAT_CSV_PATH        = os.path.join(RESULTS_DIR, "structure_resolution_statistics.csv")

# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_model(run_dir: str):
    """Load a trained ReaxModule from a checkpoint directory."""
    config_path = os.path.join(run_dir, "config.yaml")
    ckpt_path   = os.path.join(run_dir, "checkpoints", "last.ckpt")
    cfg     = omegaconf.OmegaConf.load(config_path)
    module  = hydra.utils.instantiate(cfg.model, _convert_="object")
    ckpt    = reax.training.get_default_checkpointing().load(ckpt_path)
    module.set_parameters(ckpt["parameters"])
    # Checkpoint parameters are numpy arrays; convert to JAX so jit/jacfwd work.
    params  = jax.tree_util.tree_map(jnp.asarray, module.parameters())
    return module, params, cfg


def make_graph_builder(r_max: float):
    """Return a function ase.Atoms → jraph.GraphsTuple with the same keys as
    Qm9NmrDataset.  The neighbor list is rebuilt from scratch every call, which
    is required whenever positions change."""
    return functools.partial(
        gcnn.atomic.graph_from_ase,
        r_max=r_max,
        atom_include_keys=("numbers", "nmr_tensors", "mu"),
        global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, atomic.TOTAL_ENERGY],
    )


def make_predictor(module, params, to_graph):
    """Return predict(atoms) → np.ndarray of shape (n_atoms, 3, 3).

    Rebuilds the graph (including the neighbor list) from scratch for every
    call so that topology changes when atom k moves far.
    _apply is JIT-compiled so the same topology (same n_nodes/n_edges shape)
    reuses the cached compilation.
    """
    @jax.jit
    def _apply(graph):
        return module._model.apply(params, graph)

    def predict(atoms):
        graph   = to_graph(atoms)
        out     = _apply(graph)
        n_atoms = int(graph.n_node[0])
        return np.array(out.nodes["predicted_nmr_tensors"][:n_atoms])

    return predict


def perturb_atom(atoms, atom_k, sigma, min_dist, rng, max_attempts=200):
    """Displace atom k by Gaussian noise; reject if too close to any other atom.

    Returns (perturbed_atoms, perturbed_pos_k).
    Raises RuntimeError if no valid perturbation found after max_attempts.
    """
    pos_orig = atoms.positions.copy()
    other_pos = np.delete(pos_orig, atom_k, axis=0)

    for _ in range(max_attempts):
        noise    = rng.normal(scale=sigma, size=3)
        new_pos  = pos_orig[atom_k] + noise
        dists    = np.linalg.norm(other_pos - new_pos, axis=1)
        if dists.min() >= min_dist:
            atoms_new = atoms.copy()
            atoms_new.positions[atom_k] = new_pos
            return atoms_new, new_pos

    raise RuntimeError(
        f"Could not find a valid perturbation after {max_attempts} attempts. "
        f"Try reducing sigma ({sigma} Å) or min_dist ({min_dist} Å)."
    )


def select_atom_k(atoms, n_atoms, atom_k_cfg, rng):
    """Return an atom index based on ATOM_K config.

    - None  → pick a random atom
    - int   → use that index directly
    - str   → pick a random atom of that chemical species; returns None if not found
    """
    if atom_k_cfg is None:
        return int(rng.integers(0, n_atoms))
    if isinstance(atom_k_cfg, str):
        symbols    = atoms.get_chemical_symbols()[:n_atoms]
        candidates = [i for i, s in enumerate(symbols) if s == atom_k_cfg]
        if not candidates:
            return None
        return int(rng.choice(candidates))
    return int(atom_k_cfg)


class SpeciesSpectrum(typing.NamedTuple):
    """One fitted nucleus: which atoms it covers, its ppm axis and its target curve.

    One of these per entry of `SPECTRUM_SPECIES`. They are built once from the target
    tensors and then held fixed for the whole optimisation, so the axis and the target
    never move under the optimiser's feet.
    """

    symbol: str          # "C", "H", …
    weights: jnp.ndarray  # (n_atoms,) 1.0 on this species, 0 elsewhere
    grid: jnp.ndarray     # (GRID_POINTS,) ppm axis
    target: jnp.ndarray   # (GRID_POINTS,) target spectrum


def species_spectrum(tensors, spec: SpeciesSpectrum):
    """Spectrum of one nucleus from a full set of per-atom tensors.

    `spec.weights` selects the species, so the tensors of every other element are
    multiplied by zero rather than sliced out — slicing would break the traced shapes.
    """
    return nmr_spectra.mas_spectrum(
        tensors,
        spec.grid,
        fwhm=FWHM,
        weights=spec.weights,
        lineshape=LINESHAPE,
        slope=REF_SLOPE,
        intercept=REF_INTERCEPT,
    )


def spectral_loss(tensors, specs):
    """Mean spectral mismatch over the fitted nuclei.

    Each nucleus is a separate experiment, so the losses are averaged, not the spectra.
    Pure JAX, hence usable both inside the traced optimiser loss and on the host for
    the sanity checks.
    """
    total = 0.0
    for spec in specs:
        total = total + nmr_spectra.spectrum_loss(
            species_spectrum(tensors, spec),
            spec.target,
            x=spec.grid,
            normalise_mode=NORMALISE,
            metric=METRIC,
        )
    return total / len(specs)


def build_species_spectra(atoms, n_atoms, target_tensors, species):
    """Build one `SpeciesSpectrum` per requested nucleus from the target tensors.

    The grid spans the target's own peaks padded by `GRID_MARGIN` on both sides, so a
    line that drifts during the optimisation stays inside the window (outside it the
    loss stops being meaningful — see the note in the CONFIGURATION block).
    Species that the molecule does not contain are skipped with a warning.
    """
    symbols = np.array(atoms.get_chemical_symbols()[:n_atoms])
    target_tensors = np.asarray(target_tensors)
    specs = []

    for symbol in species:
        mask = symbols == symbol
        if not mask.any():
            print(f"  ⚠ no {symbol} in this molecule — skipping its spectrum")
            continue

        iso = np.trace(target_tensors[mask], axis1=1, axis2=2) / 3.0
        shifts = REF_SLOPE * iso + REF_INTERCEPT
        grid = nmr_spectra.make_grid(
            float(shifts.min()) - GRID_MARGIN,
            float(shifts.max()) + GRID_MARGIN,
            GRID_POINTS,
        )
        weights = jnp.asarray(mask.astype(np.float32))
        spec = SpeciesSpectrum(symbol, weights, grid, None)
        target = species_spectrum(jnp.asarray(target_tensors), spec)
        specs.append(spec._replace(target=target))

        print(f"  {symbol}: {int(mask.sum())} site(s), δ ∈ [{shifts.min():.1f}, "
              f"{shifts.max():.1f}] ppm, grid [{float(grid[0]):.1f}, {float(grid[-1]):.1f}]")

    if not specs:
        sys.exit(f"None of SPECTRUM_SPECIES={species} occurs in this molecule.")
    return specs


def compute_loss(pred_tensors, specs):
    """Host-side spectral loss, for the STEP 3 sanity check and the prints.

    Identical formula to the traced loss the optimiser minimises, so the numbers in
    the log and in the plots are directly comparable.
    """
    return float(spectral_loss(jnp.asarray(pred_tensors), specs))


def save_gif(traj, atoms_orig, positions, pos_true, pos_pert,
             n_atoms, atom_k, displacement, loss_at_true, spec,
             gif_path, max_frames=120, fps=15):
    """Save an animated GIF of the optimisation trajectory.

    Layout (2×2): left column = 3-D trajectory of atom k over the predicted vs target
    SPECTRUM of the first fitted nucleus, animated in sync; right column = loss curve
    and distance-to-true curve.
    """
    pos_hist    = np.array(traj["positions"])   # (n_evals, 3)
    loss_curve  = np.array(traj["loss"])
    spectra_hist = np.array(traj["spectra"])    # (n_evals, GRID_POINTS)
    dists_hist  = np.linalg.norm(pos_hist - pos_true, axis=1)
    n_evals     = len(loss_curve)

    # Subsample to at most max_frames
    if n_evals > max_frames:
        frame_idxs = np.round(np.linspace(0, n_evals - 1, max_frames)).astype(int)
    else:
        frame_idxs = np.arange(n_evals)

    p_mean  = positions.mean(axis=0)
    _pos    = positions - p_mean
    _anum   = np.array(atoms_orig.get_atomic_numbers())[:n_atoms]
    color   = MODEL_COLOR

    # 2×2 layout: left column = 3-D path (top) + spectra (bottom);
    # right column = loss (top) + distance (bottom), stacked so there's no empty space.
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    gs  = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.30)
    ax3d    = fig.add_subplot(gs[0, 0], projection="3d")
    ax_spec = fig.add_subplot(gs[1, 0])                     # spectra, under the 3-D path
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])                     # distance, under the loss

    # ── static 3-D elements: atoms coloured by species, no bonds ──────────────
    symbols = np.array(atoms_orig.get_chemical_symbols()[:n_atoms])
    for sp in dict.fromkeys(symbols):          # unique species, first-seen order
        m = symbols == sp
        z = int(_anum[m][0])
        ax3d.scatter(_pos[m, 0], _pos[m, 1], _pos[m, 2],
                     s=_ATOM_SIZE.get(z, 40), c=_ATOM_COLOR.get(z, "violet"),
                     edgecolors="#555555", linewidths=0.3, alpha=0.85,
                     depthshade=True, label=sp)
    p_true_c = pos_true - p_mean
    p_pert_c = pos_pert - p_mean
    ax3d.scatter(*p_true_c, s=200, c="limegreen", marker="*", zorder=5,
                 label="true", edgecolors="k", linewidths=0.4)
    ax3d.scatter(*p_pert_c, s=80,  c="orangered", marker="^", zorder=5,
                 label="start", edgecolors="k", linewidths=0.4)
    ax3d.legend(fontsize=6, loc="upper left")
    ax3d.set_title(f"atom k={atom_k} trajectory", fontsize=8)
    ax3d.tick_params(labelsize=5)

    # ── animated elements ─────────────────────────────────────────────────────
    trail_line, = ax3d.plot([], [], [], "-", color=color, alpha=0.5, linewidth=1.0)
    cur_pt,     = ax3d.plot([], [], [], "o", color=color, markersize=9, zorder=6,
                             markeredgecolor="k", markeredgewidth=0.5)

    # loss panel
    loss_line, = ax_loss.plot([], [], color=color, linewidth=1.5)
    ax_loss.axhline(loss_at_true, color="limegreen", linestyle="--",
                    linewidth=1.0, label="loss @ true pos")
    ax_loss.set_xlim(0, n_evals)
    # Ignore non-finite losses (a step can land on a geometry `geometry_is_valid`
    # rejects, which scores inf) when setting limits.
    finite = loss_curve[np.isfinite(loss_curve) & (loss_curve > 0)]
    ymin = max(finite.min() * 0.9, 1e-6) if finite.size else 1e-6
    ymax = finite.max() * 1.15 if finite.size else 1.0
    ax_loss.set_ylim(ymin, ymax)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Evaluation #", fontsize=8)
    ax_loss.set_ylabel(f"spectral loss ({METRIC})", fontsize=8)
    ax_loss.set_title("Loss curve", fontsize=8)
    ax_loss.legend(fontsize=7)
    ax_loss.tick_params(labelsize=7)

    # distance panel
    dist_line, = ax_dist.plot([], [], color=color, linewidth=1.5)
    ax_dist.axhline(0.0, color="limegreen", linestyle="--",
                    linewidth=1.0, label="true pos")
    ax_dist.set_xlim(0, n_evals)
    _dmax = dists_hist[np.isfinite(dists_hist)]
    ax_dist.set_ylim(-0.02, (_dmax.max() * 1.15) if _dmax.size else 1.0)
    # ax_dist.set_yscale("log")
    ax_dist.set_xlabel("Evaluation #", fontsize=8)
    ax_dist.set_ylabel("|pos_k − pos_true| (Å)", fontsize=8)
    ax_dist.set_title("Distance to true pos", fontsize=8)
    ax_dist.legend(fontsize=7)
    ax_dist.tick_params(labelsize=7)

    # spectra panel (under the 3-D path): green = target (static), model colour =
    # predicted spectrum at the current step (animated). This is the quantity the
    # optimiser actually sees — the 3-D path above is ground truth it has no access to.
    grid = np.asarray(spec.grid)
    target_y = np.asarray(spec.target)
    # Forward-fill non-finite spectra (a line-search probe can land on a degenerate
    # geometry where the model returns inf/nan) so the animation never breaks.
    spectra_hist = spectra_hist.copy()
    last_ok = target_y
    for _t in range(len(spectra_hist)):
        if np.all(np.isfinite(spectra_hist[_t])):
            last_ok = spectra_hist[_t]
        else:
            spectra_hist[_t] = last_ok

    ax_spec.plot(grid, target_y, color="limegreen", linewidth=1.6, label="target")
    pred_line, = ax_spec.plot(grid, spectra_hist[frame_idxs[0]], color=color,
                              linewidth=1.4, label="predicted")
    ax_spec.invert_xaxis()          # NMR convention: shift decreases to the right
    _ymax = max(float(target_y.max()), float(np.nanmax(spectra_hist))) * 1.15
    ax_spec.set_ylim(0.0, _ymax if np.isfinite(_ymax) and _ymax > 0 else 1.0)
    ax_spec.set_xlabel("δ (ppm)", fontsize=8)
    ax_spec.set_ylabel("intensity", fontsize=8)
    ax_spec.set_title(f"{spec.symbol} spectrum  (green=target, colour=pred)", fontsize=8)
    ax_spec.legend(fontsize=7)
    ax_spec.tick_params(labelsize=7)

    step_txt = ax_loss.text(0.97, 0.95, "", transform=ax_loss.transAxes,
                             ha="right", va="top", fontsize=7, color="gray")

    fig.suptitle(
        f"NMR structure resolution  –  mol {MOL_IDX}  atom k={atom_k}"
        f"  |Δ|={displacement:.2f} Å",
        fontsize=9,
    )

    def _update(fi):
        i = frame_idxs[fi]
        pts = pos_hist[:i + 1] - p_mean
        trail_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        p = pts[-1]
        cur_pt.set_data_3d([p[0]], [p[1]], [p[2]])
        xs = np.arange(i + 1)
        loss_line.set_data(xs, loss_curve[:i + 1])
        dist_line.set_data(xs, dists_hist[:i + 1])
        step_txt.set_text(f"step {i + 1}/{n_evals}  loss={loss_curve[i]:.4f}")
        pred_line.set_ydata(spectra_hist[i])
        return trail_line, cur_pt, loss_line, dist_line, step_txt, pred_line

    anim = FuncAnimation(fig, _update, frames=len(frame_idxs),
                         interval=1000 // fps, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_statistics_figure(s, stat_meta, threshold, target_label, optimiser, path):
    """4-panel summary of the multi-molecule recovery test.

    Panels:
      (1) cumulative success curve – fraction of cases with |Δr| ≤ x (ECDF).
      (2) recovery vs perturbation – final |Δr| against the initial |Δr| (dist0),
          coloured by success; below the y=x line means the atom moved closer.
      (3) identifiability          – loss vs |Δr|; low loss + high |Δr| = degenerate.
      (4) |Δr| distribution        – histogram with the threshold and success rate.
    """
    if s["total"] == 0:
        print("  [stats figure] no samples — skipping.")
        return
    dist    = np.asarray(s["dist"])
    loss    = np.asarray(s["loss"])
    dist0   = np.asarray(stat_meta["dist0"])
    n_cases = len(dist)
    rec     = dist < threshold
    sr      = rec.mean()
    floor   = 1e-3   # Å – clip for log axes (perfect recoveries land here)
    C_MAIN, C_OK, C_BAD = "royalblue", "tab:green", "tab:red"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    ax_ecdf, ax_rec = axes[0]
    ax_id,   ax_hist = axes[1]

    # ── (1) cumulative success curve ──────────────────────────────────────────
    d = np.sort(dist)
    y = np.arange(1, n_cases + 1) / n_cases
    ax_ecdf.step(np.concatenate([[0.0], d]), np.concatenate([[0.0], y]),
                 where="post", color=C_MAIN, linewidth=2)
    ax_ecdf.axvline(threshold, color="k", linestyle="--", linewidth=1,
                    label=f"threshold {threshold} Å")
    ax_ecdf.set_xlabel("|Δr| threshold (Å)")
    ax_ecdf.set_ylabel("fraction recovered  (|Δr| ≤ threshold)")
    ax_ecdf.set_title(f"Cumulative success curve  (n={n_cases})")
    ax_ecdf.set_ylim(0, 1.02)
    ax_ecdf.legend(fontsize=8, loc="lower right")
    ax_ecdf.grid(alpha=0.3)

    # ── (2) recovery vs initial perturbation ──────────────────────────────────
    x0d = np.clip(dist0, floor, None)
    yfd = np.clip(dist, floor, None)
    hi  = max(x0d.max(), yfd.max()) * 1.4
    ax_rec.scatter(x0d[~rec], yfd[~rec], s=28, alpha=0.6, color=C_BAD,
                   edgecolors="k", linewidths=0.3, label="missed")
    ax_rec.scatter(x0d[rec], yfd[rec], s=28, alpha=0.7, color=C_OK,
                   edgecolors="k", linewidths=0.3, label="recovered")
    ax_rec.plot([floor, hi], [floor, hi], "k--", linewidth=1, label="y = x (no move)")
    ax_rec.axhline(threshold, color="limegreen", linestyle=":", linewidth=1)
    ax_rec.set_xscale("log"); ax_rec.set_yscale("log")
    ax_rec.set_xlim(floor, hi); ax_rec.set_ylim(floor, hi)
    ax_rec.set_xlabel("initial |Δr| (perturbation, Å)")
    ax_rec.set_ylabel("final |Δr| (Å)")
    ax_rec.set_title("Recovery vs perturbation size")
    ax_rec.legend(fontsize=8, loc="upper left")
    ax_rec.grid(alpha=0.3, which="both")

    # ── (3) identifiability: loss vs |Δr| ─────────────────────────────────────
    ax_id.scatter(yfd, np.clip(loss, 1e-6, None), s=28, alpha=0.6, color=C_MAIN,
                  edgecolors="k", linewidths=0.3)
    ax_id.axvline(threshold, color="k", linestyle="--", linewidth=1,
                  label=f"threshold {threshold} Å")
    ax_id.set_xscale("log"); ax_id.set_yscale("log")
    ax_id.set_xlabel("|Δr| final (Å)")
    ax_id.set_ylabel("loss final (RMSE, ppm)")
    ax_id.set_title("Identifiability: loss vs position error")
    ax_id.legend(fontsize=8)
    ax_id.grid(alpha=0.3, which="both")

    # ── (4) |Δr| distribution ─────────────────────────────────────────────────
    ax_hist.hist(dist, bins=min(30, max(10, n_cases // 3)), color=C_MAIN, alpha=0.75,
                 edgecolor="white")
    ax_hist.axvline(threshold, color="k", linestyle="--", linewidth=1.2,
                    label=f"threshold {threshold} Å")
    ax_hist.set_xlabel("|Δr| final (Å)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Position-error distribution  –  success {sr * 100:.0f}% "
                      f"({rec.sum()}/{n_cases})")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Structure-resolution statistics  –  {n_cases} cases  |  "
        f"target={target_label}  |  optimiser={optimiser}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Statistics figure saved to {path}")


# ── STEP 0: Load model ─────────────────────────────────────────────────────────

if not os.path.isdir(RUN_DIR):
    sys.exit(f"Model directory not found: {RUN_DIR}")

print(f"Loading model from {os.path.basename(RUN_DIR)} …")
module, params, cfg = load_model(RUN_DIR)
R_MAX     = float(cfg.data.r_max)
to_graph  = make_graph_builder(R_MAX)
predictor = make_predictor(module, params, to_graph)
print("  loaded.")
print(f"  neighbour list: {'recomputed every evaluation (jax_neighbours)' if LIVE_NEIGHBOURS else 'FROZEN at each run start'}")

# ── STEP 0b: Reproduce (part of) the run's train/val/test split ────────────────
# reax.data.random_split draws from the Engine's default nnx.Rngs stream, which REAX
# always seeds as nnx.Rngs(0) unless a caller explicitly overrides it (nothing in this
# repo does). Qm9NmrDataModule.load_split() replays that same split computation using
# only the file COUNT (cheap: no parsing) and then extracts/parses just the requested
# split's `.log` files — no saved seed or index list required, and no need to pay for
# extracting the other splits, as long as the underlying data files (and `limit`) are
# unchanged.
if SPLIT not in ("train", "val", "test", "full"):
    sys.exit(f"SPLIT must be one of 'train'/'val'/'test'/'full', got {SPLIT!r}")

print(f"\nLoading the '{SPLIT}' split from {os.path.basename(RUN_DIR)}'s config …")
extra = dict(data_dir=DATA_DIR)
if LIMIT is not None:
    extra["limit"] = LIMIT
datamodule = hydra.utils.instantiate(cfg.data, **extra, _convert_="object")

if SPLIT == "full":
    active_dataset = qm9_nmr.Qm9NmrDataset(
        r_max=datamodule._rmax,
        data_dir=datamodule._data_dir,
        dataset=datamodule._dataset,
        atom_keys=datamodule._atom_keys,
        limit=datamodule._limit,
    )
else:
    active_dataset = datamodule.load_split(SPLIT, limit=SPLIT_LIMIT)
print(f"  {len(active_dataset)} molecules in the '{SPLIT}' split.")

# ── STEP 1: Ground truth ───────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

atoms_orig  = active_dataset._data[MOL_IDX]   # ASE Atoms (original geometry)
graph_orig  = active_dataset[MOL_IDX]          # jraph.GraphsTuple (original)
n_atoms     = int(graph_orig.n_node[0])
positions   = np.array(graph_orig.nodes["positions"])   # (n_atoms, 3)

print(f"\n[{SPLIT}] molecule {MOL_IDX}: {n_atoms} atoms")
print(f"  species: {[atoms_orig.get_chemical_symbols()[i] for i in range(n_atoms)]}")

# Choose atom k
atom_k = select_atom_k(atoms_orig, n_atoms, ATOM_K, rng)
if atom_k is None:
    avail = sorted(set(atoms_orig.get_chemical_symbols()[:n_atoms]))
    sys.exit(f"No atom of species '{ATOM_K}' in molecule {MOL_IDX}. Available: {avail}")
print(f"  target atom k = {atom_k} ({atoms_orig.get_chemical_symbols()[atom_k]})")

# Ground-truth tensors
dft_tensors = np.array(graph_orig.nodes["nmr_tensors"][:n_atoms])  # (n_atoms, 3, 3)

# Model prediction at the original (true) geometry (for STEP 1 prints)
model_tensors_orig = predictor(atoms_orig)  # (n_atoms, 3, 3)

# Select target:
#   "dft"   → DFT label tensors.
#   "model" → SELF-consistency: the model's OWN prediction at the true geometry,
#             so the loss is exactly 0 at the true position (tests the landscape
#             independently of the model-vs-DFT accuracy).
if LOSS_TARGET == "dft":
    target_tensors = dft_tensors
    target_label   = "DFT label"
else:
    target_tensors = model_tensors_orig
    target_label   = "model's OWN prediction @ original geometry (self-consistency)"

print(f"\n[STEP 1] Ground truth source: {target_label}")
print(f"  DFT σ[k] (isotropic avg) = {dft_tensors[atom_k].trace()/3:.2f} ppm")
print(f"  Model σ[k] (isotropic avg) = {model_tensors_orig[atom_k].trace()/3:.2f} ppm")
print(f"  DFT vs model RMSE (all atoms) = "
      f"{np.sqrt(np.mean((dft_tensors - model_tensors_orig)**2)):.4f} ppm")

# Target SPECTRA — one per fitted nucleus. Note how little of the tensor survives:
# 9 numbers per atom become one isotropic value per site, then a single curve.
print(f"\n  Building target spectra ({METRIC} loss, fwhm={FWHM} ppm, "
      f"lineshape={LINESHAPE}):")
specs = build_species_spectra(atoms_orig, n_atoms, target_tensors, SPECTRUM_SPECIES)
n_sites = int(sum(float(s.weights.sum()) for s in specs))
print(f"  → fitting {len(specs)} spectrum/spectra over {n_sites} site(s); "
      f"atom k is {'' if any(float(s.weights[atom_k]) > 0 for s in specs) else 'NOT '}"
      f"among them")

# ── STEP 2: Perturbation ───────────────────────────────────────────────────────

print(f"\n[STEP 2] Perturbing atom {atom_k} with σ={SIGMA} Å …")
atoms_pert, pos_pert = perturb_atom(
    atoms_orig, atom_k, SIGMA, MIN_DIST_PERTURB, rng, MAX_PERTURB_ATTEMPTS
)
pos_true = positions[atom_k].copy()
displacement = np.linalg.norm(pos_pert - pos_true)
print(f"  |Δpos| = {displacement:.4f} Å")
print(f"  true pos:      {pos_true}")
print(f"  perturbed pos: {pos_pert}")

# ── STEP 3: Loss function + sanity check ───────────────────────────────────────

print(f"\n[STEP 3] Building loss function …")


def trajectory_spectra(module, params, atom_k, specs, atoms_base, positions,
                       rebuild=None, chunk=16):
    """Predicted spectrum of the first fitted nucleus at each of `positions`.

    The optimiser no longer records spectra as it goes — with `N_STARTS` runs in flight
    that would be an (N_STARTS, ADAM_STEPS, GRID_POINTS) array — so the GIF's curves are
    recomputed afterwards, for the winning trajectory only.

    Batched in chunks of `chunk`: vmapping a whole 500-step trajectory at once asks XLA
    for ~7 GB of activations and dies. One compilation is shared by every full chunk; a
    ragged tail is padded up to `chunk` rather than triggering a second one.
    """
    n_atoms = specs[0].weights.shape[0]
    graph_base = to_graph(atoms_base)
    pos_all = jnp.asarray(atoms_base.positions, dtype=jnp.float32)

    @jax.jit
    @jax.vmap
    def _spec(pos_k):
        pos = pos_all.at[atom_k].set(pos_k)
        g = gcnn.experimental.update_graph(graph_base).set(("nodes", "positions"), pos).get()
        if rebuild is not None:
            g = rebuild(g, pos)
        out = module._model.apply(params, g)
        return species_spectrum(out.nodes["predicted_nmr_tensors"][:n_atoms], specs[0])

    pos_arr = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    out = []
    for i in range(0, len(pos_arr), chunk):
        block = pos_arr[i:i + chunk]
        pad = chunk - len(block)
        if pad:
            block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
        res = np.asarray(_spec(jnp.asarray(block)))
        out.append(res[:chunk - pad] if pad else res)
    return np.concatenate(out, axis=0)



_RESTART_RNG = np.random.default_rng(SEED + 1)   # seeds the multistart (reproducible)


def optimize_position(module, params, to_graph, atoms_base, atom_k,
                      specs, x0, pos_true):
    """Recover atom k's position: `N_STARTS` parallel Adam descents, best observed loss wins.

    `x0` is the prior on where the atom is — the perturbed position in this benchmark, a
    candidate site in a real refinement. `N_STARTS` starts are drawn in a ball of
    `START_RADIUS` around it and descended simultaneously by a vmapped Adam (see
    `make_multistart_adam`); each stays within `LOCAL_CAP` of its own start, so coverage
    comes from the spread of the starts rather than from any single run roaming.

    `pos_true` is used ONLY for printing and for the returned diagnostics. It never enters
    the selection, which sees the loss alone — the same information a real experiment gives.

    Returns dict(pos, loss, traj, n_evals, n_starts, n_valid, alternatives, hits).
    """
    x0_arr = np.asarray(x0, dtype=np.float64).reshape(3)
    pos_true_arr = np.asarray(pos_true, dtype=np.float64).reshape(3)
    other_pos = np.delete(np.asarray(atoms_base.positions, dtype=np.float64), atom_k, axis=0)
    cfg_s = ss.SearchConfig(
        adam_steps=ADAM_STEPS, adam_lr0=ADAM_LR0, adam_lr1=ADAM_LR1,
        local_cap=LOCAL_CAP if LOCAL_CAP is not None else 1e9,
        min_dist_guard=MIN_DIST_GUARD, guard_margin=GUARD_MARGIN, guard_weight=GUARD_WEIGHT,
        starts_per_batch=STARTS_PER_BATCH, dist_threshold=DIST_THRESHOLD,
        degeneracy_sep=DEGENERACY_SEP, degeneracy_tol=DEGENERACY_TOL)
    loss_fn = lambda pred: spectral_loss(pred, specs)   # ties the engine to the spectra

    # The prior itself is always one of the starts, so the multistart can never do worse
    # than a single run from the guess.
    starts = ss.sample_ball_starts(x0_arr, other_pos, N_STARTS, START_RADIUS, cfg_s,
                                   _RESTART_RNG)
    starts[0] = x0_arr

    # One rebuilder per structure (captures the slot capacity, fixed for the run). Size the
    # capacity against a spread of the starts so it covers where the atom will roam.
    rebuild = None
    if LIVE_NEIGHBOURS:
        probes = [np.where(np.arange(len(atoms_base))[:, None] == atom_k, s, atoms_base.positions)
                  for s in starts[:min(16, len(starts))]]
        rebuild, _capacity = ss.make_neighbour_rebuilder(
            atoms_base, R_MAX, NEIGHBOUR_CAPACITY, probe_positions=probes)

    out = ss.run_multistart(module, params, to_graph, atoms_base, atom_k, loss_fn, starts,
                            cfg_s, rebuild=rebuild)
    losses = out["losses"]
    positions_out = out["positions"]
    all_losses = out["all_losses"]   # (N_STARTS, ADAM_STEPS)
    all_pos = out["all_pos"]         # (N_STARTS, ADAM_STEPS, 3)
    finite = np.where(np.isfinite(losses), losses, np.inf)
    n_valid = int(np.isfinite(finite).sum())
    if n_valid == 0:
        raise RuntimeError(
            "every start ended on an invalid geometry — check MIN_DIST_GUARD/START_RADIUS."
        )
    i_best = int(np.argmin(finite))

    # Trajectory of the winning start, for the plots and the GIF.
    traj = dict(
        positions=[np.asarray(p, dtype=np.float64) for p in np.asarray(all_pos)[i_best]],
        loss=[float(v) for v in np.asarray(all_losses)[i_best]],
        spectra=[],   # filled in on demand by the caller (one forward pass per GIF frame)
    )

    # Ground-truth-free degeneracy check, plus a ground-truth diagnostic kept apart from it.
    clusters = ss.cluster_minima(finite, positions_out, cfg_s, cell=None)
    alternatives = ss.find_degenerate_alternatives(clusters, cfg_s, cell=None)
    hits = int((np.linalg.norm(positions_out - pos_true_arr, axis=1) < DIST_THRESHOLD).sum())

    if PRINT_EVALS:
        order = np.argsort(finite)[:10]
        for rank, i in enumerate(order):
            print(f"      start {int(i):3d} (rank {rank}): loss {finite[i]:8.3f}  "
                  f"|Δr| {np.linalg.norm(positions_out[i] - pos_true_arr):.4f} Å")

    rmsd_b = float(np.linalg.norm(positions_out[i_best] - pos_true_arr))
    rmsd0 = float(np.linalg.norm(x0_arr - pos_true_arr))
    # `starts[0]` is the prior guess itself, so this is the loss the search started from —
    # not `traj["loss"][0]`, which is only where the WINNING start happened to begin.
    loss_at_guess = float(all_losses[0][0])
    print(f"    {METHOD_LABEL}: {N_STARTS} starts × {ADAM_STEPS} steps  "
          f"loss {loss_at_guess:8.3f} → {finite[i_best]:8.3f}  "
          f"|Δr| {rmsd0:.4f} → {rmsd_b:.4f} Å  "
          f"({hits}/{N_STARTS} starts reached the true basin)", flush=True)
    if alternatives:
        alt_txt = ", ".join(f"{d:.2f} Å (loss {l:.3f})" for d, l in alternatives[:3])
        print(f"    ⚠ DEGENERATE: {len(alternatives)} distinct position(s) fit within "
              f"{100 * DEGENERACY_TOL:.0f}% of the best loss — {alt_txt}")
        print(f"      the spectrum does not determine this site; treat the answer as "
              f"unresolved rather than correct.")

    return dict(pos=positions_out[i_best], loss=float(finite[i_best]), traj=traj,
                loss_at_guess=loss_at_guess, n_evals=N_STARTS * ADAM_STEPS,
                n_starts=N_STARTS, n_valid=n_valid,
                alternatives=alternatives, hits=hits, rebuild=rebuild)


# Sanity check
print("  Sanity check:")
pred_true = predictor(atoms_orig)
pred_pert = predictor(atoms_pert)
loss_at_true = compute_loss(pred_true, specs)
l_pert       = compute_loss(pred_pert, specs)
print(f"    loss @ true position  = {loss_at_true:.6f}")
print(f"    loss @ perturbed pos  = {l_pert:.6f}")
# The landscape is only usable if perturbing the atom actually RAISES the loss.
# If loss@perturbed is not clearly above loss@true, the NMR tensors of atom k +
# neighbours barely depend on k's position → the landscape is flat and no
# optimiser can pull the atom back (this is physics, not a code bug).
#   - "dft"  mode: hard requirement (loss@true > 0), so we assert.
#   - "model" mode: loss@true ≈ 0 by construction, so we only WARN.
if l_pert <= loss_at_true * 1.05:
    msg = (f"loss@perturbed ({l_pert:.6f}) is not clearly above "
           f"loss@true ({loss_at_true:.6f}) → flat landscape, recovery unlikely.")
    if LOSS_TARGET != "model":
        raise AssertionError(f"Sanity check FAILED: {msg}")
    print(f"    ⚠ Sanity check WEAK: {msg}")
else:
    print(f"    Sanity check PASSED ✓  (Δloss = {l_pert - loss_at_true:+.6f})")

# ── STEP 4: Optimisation ───────────────────────────────────────────────────────

print(f"\n[STEP 4] Optimising ({METHOD_LABEL}) starting from perturbed position …")

x0   = pos_pert.copy().flatten()
best = optimize_position(
    module, params, to_graph, atoms_orig, atom_k, specs, x0, pos_true,
)
traj = best["traj"]

pos_recovered = np.asarray(best["pos"]).reshape(3)
rmsd          = float(np.linalg.norm(pos_recovered - pos_true))
# REAL success = the atom is back near its true position (positional |Δr| in Å),
# independent of any optimiser convergence flag.
recovered  = bool(rmsd < DIST_THRESHOLD)
n_evals    = best["n_evals"]
loss_init  = best["loss_at_guess"]
loss_final = float(best["loss"])

print(f"    optimiser: {METHOD_LABEL}  ({n_evals} gradient evaluations total, "
      f"{best['n_valid']}/{best['n_starts']} starts ended on a valid geometry)")
print(f"    loss: {loss_init:.6f} → {loss_final:.6f}")
print(f"    |Δr|(recovered, true) = {rmsd:.4f} Å  (started at {displacement:.4f} Å)")
print(f"    >>> RECOVERED: {recovered}  "
      f"(|Δr| {'<' if recovered else '≥'} {DIST_THRESHOLD} Å threshold)")
print(f"    recovered pos: {pos_recovered}")
# Benchmark-only cross-check, impossible in a real refinement: is the answer wrong
# because the SEARCH failed, or because the model's minimum is not at the true geometry?
# A recovered loss at or below loss@true while |Δr| is large means the latter, and no
# amount of extra starts will fix it.
if not recovered:
    if loss_final <= loss_at_true * (1.0 + DEGENERACY_TOL):
        print(f"    → the model prefers this wrong position: loss {loss_final:.3f} vs "
              f"{loss_at_true:.3f} at the TRUE geometry. Model-accuracy limit, not a "
              f"search failure.")
    else:
        print(f"    → search failure: loss {loss_final:.3f} is still well above "
              f"{loss_at_true:.3f} at the true geometry. More starts should help.")

# Topology self-check. The optimiser minimises the loss on ITS OWN graph: a neighbour
# list recomputed at every evaluation (LIVE_NEIGHBOURS) or one frozen at the start of the
# run. This recomputes the same loss from a list rebuilt on the host at the recovered
# geometry — the ground truth for the topology. A large gap means the optimiser was
# descending the wrong landscape.
atoms_check = atoms_orig.copy()
atoms_check.positions[atom_k] = pos_recovered
loss_rebuilt = compute_loss(predictor(atoms_check), specs)
_gap = abs(loss_final - loss_rebuilt)
print(f"    topology check: optimiser's loss {loss_final:.6f}  vs  rebuilt-list "
      f"{loss_rebuilt:.6f}   (gap {_gap:.2e})")
if _gap > 0.05 * max(abs(loss_rebuilt), 1e-12):
    print(f"    ⚠ the optimiser's neighbour list disagrees with a rebuilt one by "
          f"{100 * _gap / max(abs(loss_rebuilt), 1e-12):.0f}% — set LIVE_NEIGHBOURS=True "
          f"(currently {LIVE_NEIGHBOURS}) or raise NEIGHBOUR_CAPACITY.")

# ── STEP 5: Evaluation and plots ──────────────────────────────────────────────

print(f"\n[STEP 5] Plotting …")

# Colour/size maps (also used by the GIF helpers below).
_ATOM_COLOR = {1: "#E8E8E8", 6: "#909090", 7: "#3050F8", 8: "#FF2010", 9: "#90E050"}
_ATOM_SIZE  = {1: 15, 6: 60, 7: 60, 8: 60, 9: 55}
MODEL_COLOR = "royalblue"

# Layout: loss | distance | one overlaid-spectra panel per fitted nucleus.
n_spec = len(specs)
fig = plt.figure(figsize=(5 * (2 + n_spec), 4), facecolor="white")
gs  = fig.add_gridspec(1, 2 + n_spec, hspace=0.45, wspace=0.35)

loss_curve = np.array(traj["loss"])

# Column 0: loss vs optimisation step
ax_loss = fig.add_subplot(gs[0, 0])
ax_loss.semilogy(np.arange(len(loss_curve)), loss_curve, color=MODEL_COLOR, linewidth=1.5)
ax_loss.axhline(loss_at_true, color="limegreen", linestyle="--",
                linewidth=1.0, label="loss @ true pos")
ax_loss.set_xlabel("Evaluation #", fontsize=8)
ax_loss.set_ylabel(f"spectral loss ({METRIC})", fontsize=8)
ax_loss.set_title("Loss curve", fontsize=9)
ax_loss.legend(fontsize=7)
ax_loss.tick_params(labelsize=7)

# Column 1: distance to true position vs step
ax_dist    = fig.add_subplot(gs[0, 1])
pos_hist   = np.array(traj["positions"])   # (n_evals, 3)
dists_hist = np.linalg.norm(pos_hist - pos_true, axis=1)
ax_dist.plot(np.arange(len(dists_hist)), dists_hist, color=MODEL_COLOR, linewidth=1.5)
ax_dist.axhline(0.0, color="limegreen", linestyle="--", linewidth=1.0,
                label="true position")
# This is the WINNING start's path, and the answer is the lowest-loss point it visited,
# not wherever the 500 Adam steps happened to end. Mark it so the panel cannot be misread
# as ending at the reported result.
ax_dist.axhline(rmsd, color="k", linestyle=":", linewidth=1.0,
                label=f"returned |Δr| = {rmsd:.3f} Å")
ax_dist.set_xlabel("Evaluation #", fontsize=8)
ax_dist.set_ylabel("|pos_k − pos_true| (Å)", fontsize=8)
ax_dist.set_title("Distance to true pos", fontsize=9)
ax_dist.legend(fontsize=7)
ax_dist.tick_params(labelsize=7)

# Remaining columns: the spectra themselves — target vs perturbed vs recovered.
# This is what the optimiser sees; the two panels on the left are ground truth it
# has no access to.
atoms_rec = atoms_orig.copy()
atoms_rec.positions[atom_k] = pos_recovered
pred_rec  = jnp.asarray(predictor(atoms_rec))
pred_pert_t = jnp.asarray(predictor(atoms_pert))

spectra_rows = []
for col, spec in enumerate(specs):
    ax = fig.add_subplot(gs[0, 2 + col])
    grid_np = np.asarray(spec.grid)
    y_tgt  = np.asarray(spec.target)
    y_pert = np.asarray(species_spectrum(pred_pert_t, spec))
    y_rec  = np.asarray(species_spectrum(pred_rec, spec))

    ax.plot(grid_np, y_tgt,  color="limegreen", linewidth=1.8, label="target")
    ax.plot(grid_np, y_pert, color="orangered", linewidth=1.2, linestyle="--",
            label="perturbed")
    ax.plot(grid_np, y_rec,  color=MODEL_COLOR, linewidth=1.4, label="recovered")
    ax.invert_xaxis()               # NMR convention: δ decreases to the right
    ax.set_xlabel("δ (ppm)", fontsize=8)
    ax.set_ylabel("intensity", fontsize=8)
    l_pert_s = float(nmr_spectra.spectrum_loss(jnp.asarray(y_pert), spec.target,
                                               x=spec.grid, normalise_mode=NORMALISE,
                                               metric=METRIC))
    l_rec_s = float(nmr_spectra.spectrum_loss(jnp.asarray(y_rec), spec.target,
                                              x=spec.grid, normalise_mode=NORMALISE,
                                              metric=METRIC))
    ax.set_title(f"{spec.symbol} spectrum   loss {l_pert_s:.3f} → {l_rec_s:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    spectra_rows.append((spec.symbol, grid_np, y_tgt, y_pert, y_rec))

fig.suptitle(
    f"NMR structure resolution  –  molecule {MOL_IDX}  atom k={atom_k}  "
    f"target={target_label}",
    fontsize=10,
)
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved to {PLOT_PATH}")

# ── CSV ───────────────────────────────────────────────────────────────────────
if SAVE_CSV:
    with open(CSV_TRAJ_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "pos_x", "pos_y", "pos_z", "dist_to_true_ang"])
        for step, (pos_k, loss_val) in enumerate(
            zip(traj["positions"], traj["loss"]), start=1
        ):
            dist = float(np.linalg.norm(pos_k - pos_true))
            w.writerow([step, f"{loss_val:.8f}",
                        f"{pos_k[0]:.6f}", f"{pos_k[1]:.6f}", f"{pos_k[2]:.6f}",
                        f"{dist:.6f}"])
    print(f"  Trajectory CSV saved to {CSV_TRAJ_PATH}")

    with open(CSV_SUMMARY_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_idx", "atom_k", "displacement_ang",
                    "n_evals", "loss_init", "loss_final", "loss_at_true", "dist_ang",
                    "recovered", "n_starts", "n_degenerate_alternatives"])
        w.writerow([MOL_IDX, atom_k, f"{displacement:.6f}",
                    n_evals, f"{loss_init:.8f}",
                    f"{loss_final:.8f}", f"{loss_at_true:.8f}", f"{rmsd:.6f}",
                    recovered, best["n_starts"], len(best["alternatives"])])
    print(f"  Summary CSV saved to {CSV_SUMMARY_PATH}")

# ── GIF ───────────────────────────────────────────────────────────────────────
if SAVE_GIF:
    # Spectra are not recorded during the descent (see `trajectory_spectra`), so the
    # animated curve is recomputed here for the winning trajectory.
    traj["spectra"] = list(trajectory_spectra(
        module, params, atom_k, specs, atoms_orig, traj["positions"],
        rebuild=best["rebuild"],
    ))
    print(f"  Saving GIF → {GIF_PATH} …")
    save_gif(
        traj, atoms_orig, positions, pos_true, pos_pert,
        n_atoms, atom_k, displacement,
        loss_at_true, specs[0],
        GIF_PATH, max_frames=GIF_MAX_FRAMES, fps=GIF_FPS,
    )
    print(f"    GIF saved.")

if SAVE_SPECTRA_CSV:
    with open(CSV_SPECTRA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "delta_ppm", "target", "perturbed", "recovered"])
        for symbol, gx, y_t, y_p, y_r in spectra_rows:
            for i in range(len(gx)):
                w.writerow([symbol, f"{gx[i]:.4f}", f"{y_t[i]:.8e}",
                            f"{y_p[i]:.8e}", f"{y_r[i]:.8e}"])
    print(f"  Spectra CSV saved to {CSV_SPECTRA_PATH}")

# ── STEP 6: Statistics (multi-atom / multi-molecule loop) ─────────────────────

if not RUN_STATS:
    print("\n[STEP 6] Skipped (RUN_STATS=False).")
else:
    print("\n[STEP 6] Statistics over multiple molecules …")
    print(f"  N_MOL_STAT={N_MOL_STAT}  N_ATOM_STAT={N_ATOM_STAT}"
          f"  ATOM_K={ATOM_K!r}  DIST_THRESHOLD={DIST_THRESHOLD} Å\n")

    rng_stat = np.random.default_rng(SEED + 100)
    # Lists are appended in case order, index-aligned with stat_meta.
    #   success  = |Δr| < DIST_THRESHOLD                       (needs ground truth)
    #   reliable = success AND loss_final < RELIABLE_LOSS_TOL   (observable-only proxy)
    # `loss_true` and `hits` need ground truth and exist only to tell a SEARCH failure
    # (loss still far above the one at the true geometry — more starts would help) from a
    # MODEL limit (loss at or below it, so the model genuinely prefers the wrong position).
    # `n_alt` is the same warning built from observables alone, and is the one that carries
    # over to a real refinement.
    stats     = dict(dist=[], loss=[], n_evals=[], n_starts=[], success=0, reliable=0,
                     total=0, model_limited=0, search_failed=0)
    stat_meta = dict(mol=[], atom=[], species=[], dist0=[], loss_true=[], hits=[], n_alt=[])

    stat_start = MOL_IDX + 1
    stat_end   = min(stat_start + N_MOL_STAT, len(active_dataset))
    for mol_i in range(stat_start, stat_end):
        atoms_i = active_dataset._data[mol_i]
        graph_i = active_dataset[mol_i]
        ni      = int(graph_i.n_node[0])
        pos_i   = np.array(graph_i.nodes["positions"])
        dft_i   = np.array(graph_i.nodes["nmr_tensors"][:ni])

        for _ in range(N_ATOM_STAT):
            k_i = select_atom_k(atoms_i, ni, ATOM_K, rng_stat)
            if k_i is None:
                continue   # species not present in this molecule
            try:
                atoms_pert_i, pos_pert_i = perturb_atom(
                    atoms_i, k_i, SIGMA, MIN_DIST_PERTURB, rng_stat, MAX_PERTURB_ATTEMPTS
                )
            except RuntimeError:
                continue

            stat_meta["mol"].append(mol_i)
            stat_meta["atom"].append(k_i)
            stat_meta["species"].append(atoms_i.get_chemical_symbols()[k_i])
            stat_meta["dist0"].append(float(np.linalg.norm(pos_pert_i - pos_i[k_i])))

            # Target spectra: from the DFT labels, or from the model's OWN
            # self-consistent prediction at the true geometry. Rebuilt per molecule,
            # since each one has its own sites and therefore its own ppm range.
            tgt_i = dft_i if LOSS_TARGET == "dft" else predictor(atoms_i)
            specs_i = build_species_spectra(atoms_i, ni, tgt_i, SPECTRUM_SPECIES)
            best_i = optimize_position(
                module, params, to_graph, atoms_i, k_i, specs_i,
                pos_pert_i.flatten(), pos_i[k_i],
            )
            dist_i = float(np.linalg.norm(np.asarray(best_i["pos"]).reshape(3) - pos_i[k_i]))
            loss_i = float(best_i["loss"])
            # Reference point for the diagnosis below: what the loss is worth at the
            # correct geometry. Benchmark-only — no experiment provides it.
            loss_true_i = compute_loss(predictor(atoms_i), specs_i)
            stats["dist"].append(dist_i)
            stats["loss"].append(loss_i)
            stats["n_evals"].append(best_i["n_evals"])
            stats["n_starts"].append(best_i["n_starts"])
            stat_meta["loss_true"].append(loss_true_i)
            stat_meta["hits"].append(best_i["hits"])
            stat_meta["n_alt"].append(len(best_i["alternatives"]))
            stats["total"] += 1
            ok_dist = dist_i < DIST_THRESHOLD
            ok_loss = loss_i < RELIABLE_LOSS_TOL
            if ok_dist:
                stats["success"] += 1
            if ok_dist and ok_loss:
                stats["reliable"] += 1
            model_limited = (not ok_dist) and loss_i <= loss_true_i * (1.0 + DEGENERACY_TOL)
            if not ok_dist:
                stats["model_limited" if model_limited else "search_failed"] += 1

            if ok_dist and ok_loss:
                flag = "✓ reliable"
            elif ok_dist:
                flag = "~ close"
            elif model_limited:
                flag = "✗ miss (model limit)"
            else:
                flag = "✗ miss (search)"
            print(f"  mol {mol_i:2d}  atom {k_i:2d} ({stat_meta['species'][-1]}) → "
                  f"|Δr|={dist_i:.3f} Å  loss={loss_i:.2f} (true {loss_true_i:.2f}) ppm  "
                  f"({best_i['hits']}/{best_i['n_starts']} starts on target, "
                  f"{len(best_i['alternatives'])} rival minima)  {flag}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    if stats["total"] == 0:
        print("  no samples")
    else:
        print(
            f"  success  rate = {stats['success']}/{stats['total']} "
            f"(|Δr| < {DIST_THRESHOLD} Å)"
        )
        print(
            f"  reliable rate = {stats['reliable']}/{stats['total']} "
            f"(|Δr| < {DIST_THRESHOLD} Å AND loss < {RELIABLE_LOSS_TOL} ppm)"
        )
        print(
            f"  median |Δr| = {np.median(stats['dist']):.3f} Å  |  "
            f"mean |Δr| = {np.mean(stats['dist']):.3f} Å  |  "
            f"mean starts = {np.mean(stats['n_starts']):.0f}"
        )
        # The two failure modes need opposite responses, so never report them as one number.
        print(
            f"  of the {stats['total'] - stats['success']} miss(es): "
            f"{stats['search_failed']} search failure(s) → raise N_STARTS/START_RADIUS;  "
            f"{stats['model_limited']} model limit(s) → the model's minimum is not at the "
            f"true geometry, only a better checkpoint helps."
        )
        _hits = np.asarray(stat_meta["hits"], dtype=float)
        _tot_starts = float(np.sum(stats["n_starts"]))
        _rate = _hits.sum() / max(_tot_starts, 1.0)
        print(
            f"  per-start hit rate = {int(_hits.sum())}/{int(_tot_starts)} = {100*_rate:.1f}%"
            f"  → P(at least one hit) at N_STARTS={N_STARTS} is "
            f"{100 * (1 - (1 - _rate) ** N_STARTS):.0f}%"
        )
        _flagged = int(np.sum(np.asarray(stat_meta["n_alt"]) > 0))
        print(
            f"  degeneracy warnings = {_flagged}/{stats['total']} case(s) had a rival "
            f"minimum within {100*DEGENERACY_TOL:.0f}% of the best loss"
        )

    # ── Statistics figure (4 panels) ──────────────────────────────────────────
    save_statistics_figure(
        stats, stat_meta, DIST_THRESHOLD, target_label, METHOD_LABEL, STAT_PLOT_PATH
    )

    # ── Per-case statistics CSV ────────────────────────────────────────────────
    if SAVE_CSV and stat_meta["mol"]:
        with open(STAT_CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "mol_idx", "atom_k", "species", "dist0_ang",
                        "dist_ang", "loss", "loss_at_true", "n_evals", "n_starts",
                        "starts_on_target", "n_rival_minima", "recovered", "reliable",
                        "model_limited"])
            for i in range(len(stat_meta["mol"])):
                d = stats["dist"][i]
                l = stats["loss"][i]
                lt = stat_meta["loss_true"][i]
                ok = d < DIST_THRESHOLD
                w.writerow([i, stat_meta["mol"][i], stat_meta["atom"][i],
                            stat_meta["species"][i], f"{stat_meta['dist0'][i]:.6f}",
                            f"{d:.6f}", f"{l:.8f}", f"{lt:.8f}",
                            stats["n_evals"][i], stats["n_starts"][i],
                            stat_meta["hits"][i], stat_meta["n_alt"][i],
                            bool(ok), bool(ok and l < RELIABLE_LOSS_TOL),
                            bool((not ok) and l <= lt * (1.0 + DEGENERACY_TOL))])
        print(f"  Statistics CSV saved to {STAT_CSV_PATH}")

print(f"\nDone.")
print(f"  Plot:    {PLOT_PATH}")
if SAVE_CSV:
    print(f"  CSV:     {CSV_TRAJ_PATH}")
    print(f"           {CSV_SUMMARY_PATH}")
if SAVE_SPECTRA_CSV:
    print(f"           {CSV_SPECTRA_PATH}")
if SAVE_GIF:
    print(f"  GIF:     {GIF_PATH}")
if RUN_STATS:
    print(f"  Stats:   {STAT_PLOT_PATH}")
    if SAVE_CSV:
        print(f"           {STAT_CSV_PATH}")
