#!/usr/bin/env python3
"""
NMR-guided structure resolution for CSH — gradient-based, fitted against SPECTRA.

Same experiment as the ``full_tensors`` variant, but the loss is the mismatch between
predicted and target *spectra* instead of per-atom tensors. This is the deployment
setting: a real measurement gives one curve per nucleus, never the per-atom tensors
and never their orientation.

The information loss is severe, and CSH makes it starker than QM9. A fast-MAS spectrum
keeps only ``σ_iso = tr(σ)/3`` per site, and a bulk CSH cell contains a HANDFUL of Na —
often exactly one. Fitting 3 coordinates against a single number cannot work, however
good the model is, which is why `SPECTRUM_SPECIES` defaults to fitting ²³Na *and* ²⁹Si
together: every nucleus in the cell responds to where the Na sits, so each extra
nucleus buys equations. Measured on a bulk test structure, displacing the Na by 0.3 Å
moves σ_iso by:

    Na  12.3 ppm (1 site)   ·  O   9.8 ppm (36 sites)  ·  Ca  2.7 ppm (10 sites)
    Si   0.4 ppm (9 sites)  ·  H   0.5 ppm (15 sites)

Note the tension: ¹⁷O is both sensitive and abundant but needs isotopic enrichment,
while ²⁹Si is the routine measurement yet responds ~30× more weakly. Adding "O" to
`SPECTRUM_SPECIES` is therefore the most informative *numerical* experiment and the
least realistic *physical* one — worth running to separate "the method is broken" from
"this observable is too poor".

Two caveats specific to this system:

- **²³Na is quadrupolar (spin 3/2).** A real ²³Na MAS lineshape carries a second-order
  quadrupolar shift and asymmetry set by the electric field gradient, which a shielding
  model does not predict. What is synthesised here is the chemical-shift contribution
  only — self-consistent against DFT labels or against the model itself, but NOT
  directly comparable to a measured ²³Na spectrum without an EFG model.
- **Periodic solid.** Neighbour lists come from the cell (`graph_from_ase`) and every
  host-side distance goes through `min_image_distances`.

Steps:
  1. Ground-truth spectra (from DFT labels or from the model at the true geometry).
  2. Gaussian perturbation of atom k with validity checks (minimum image).
  3. Loss = spectral mismatch, averaged over the fitted nuclei.
  4. Gradient-based local minimisation (gcnn.adapt + jax.scipy.optimize BFGS, 3 DOF).
  5. Evaluation: |Δr|, loss curve, overlaid spectra; per-run GIF.
  6. (Optional) Statistics over many structures.
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

from e3response import keys, nmr_spectra

logging.getLogger("reax").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Model checkpoint directory (must contain config.yaml and checkpoints/last.ckpt).
RUN_DIR = "/home/mattia/Desktop/ml_codes/nmr_project/csh_nmr_runs/bulk/csh_rmax5_heavy/nequip_nmr_heavy"

# Dataset — DATASET/R_MAX/train_val_test_split come from RUN_DIR/config.yaml (cfg.data
# below), so the reproduced split matches the run. Only data_file is overridden, since the
# run's config.yaml may store a relative/cluster path.
DATA_FILE = "/home/mattia/Desktop/ml_codes/CAMML/e3response/data/csh_nmr/Na_csh.json"
SPLIT    = "test"      # which partition to work on: "train" | "val" | "test" | "full"
LIMIT    = None       # base dataset restriction, applied BEFORE splitting (None → use the
                      # run's own `limit`, required to reproduce ITS split exactly). Same
                      # syntax as before: int N, "n:m", or "n:m:step".
SPLIT_LIMIT = None    # further restriction WITHIN the chosen SPLIT (ignored if SPLIT="full"),
                      # e.g. SPLIT_LIMIT=20 → only the first 20 test structures,
                      # "10:30" → test structures 10-29. Only extracts/parses those.

# Demo: which molecule and atom
MOL_IDX = 0      # index into the chosen limit or SPLIT and limit (not the raw dataset)
ATOM_K  = "Na"    # None → random atom; int → direct index; str → random atom of that species (e.g. "N")
SEED    = 42     # fixed seed for reproducibility

# Perturbation
SIGMA                = 0.8  # Gaussian noise amplitude (Å)
MIN_DIST             = 0.2   # min allowed distance to any other atom (Å)
MAX_PERTURB_ATTEMPTS = 200   # retries for valid perturbation

# Loss function
# "dft"   → target = spectrum built from the DFT labels in the graph
# "model" → target = spectrum of the model's OWN prediction at the ORIGINAL geometry
# Defaults to "model" here: on CSH the self-consistency test (loss@true ≡ 0) must pass
# BEFORE any DFT- or experiment-referenced result is worth interpreting, otherwise a
# failure cannot be attributed between the optimiser, the landscape and model error.
LOSS_TARGET = "model"

# ── Spectrum ──────────────────────────────────────────────────────────────────
# Which nuclei are fitted. Each is an INDEPENDENT experiment: one spectrum and one loss
# term per species, averaged — never summed into a single curve. See the module
# docstring for the measured sensitivities: ("Na",) alone is one line for one site and
# is expected to be hopeless; ("Na", "O") is the most informative and the least
# experimentally realistic. Species absent from a structure are skipped with a warning.
SPECTRUM_SPECIES = ("Na", "Si")
# Linewidth in ppm: a single float for every nucleus, or one value per species. ²³Na
# lines in CSH are broad (quadrupolar, unmodelled here); ²⁹Si MAS lines are narrower.
FWHM        = {"Na": 8.0, "Si": 3.0, "O": 8.0, "Ca": 8.0, "H": 2.0}
LINESHAPE   = "gaussian"   # "gaussian" | "lorentzian" | "pseudo_voigt"
# Metric. "wasserstein" (ppm, transport distance) is the default because point-by-point
# metrics go FLAT once peaks stop overlapping — on this system that happens after
# ~0.1 Å, leaving the optimiser with no gradient at all.
METRIC      = "wasserstein"
NORMALISE   = "area"       # "area" | "max" | "none"
# Referencing δ = slope·σ + intercept. The default mirrors the axis (δ = −σ), which is
# self-consistent since target and prediction share the convention. Comparing to REAL
# spectra requires calibrating these per nucleus at the same level of theory.
REF_SLOPE     = -1.0
REF_INTERCEPT = 0.0
# The grid must hold the peaks for the WHOLE optimiser excursion: if a line leaves the
# window its area vanishes, normalisation blows up, and both loss and gradient become
# artefacts rather than signal. Na moves ~95 ppm/Å here, so keep this generous — and
# note that MAX_DISPLACEMENT=None (no bound) lets BFGS travel arbitrarily far.
GRID_MARGIN = 150.0        # ppm of padding beyond the target's own peak range
GRID_POINTS = 1024

# Optimisation:
# gcnn.adapt + jax.scipy.optimize.minimize (BFGS). The NMR-tensor loss is
# differentiable end-to-end (the model recomputes edge vectors from positions),
# so jopt minimises it fully on device with exact autodiff gradients. 

METHOD_LABEL = "jopt-BFGS"
MAX_ITER     = 1000
GRAD_CHECK   = False   # finite-difference-verify the JAX gradient
PRINT_EVALS  = True    # print loss + |Δpos| at every optimiser evaluation (incl. line-search
                       # probes, via jax.debug.print so it works inside the on-device loop).
                       # Noisy for multi-molecule STEP 6 stats — turn off there if needed.
# Smoothly cap atom k's displacement from its start: the optimised variable u is mapped
# to pos_k = x0 + R·tanh((u − x0)/R), so |pos_k − x0| < R (Å) for ANY u. Keeps BFGS from
# firing the atom into far, degenerate geometries (inf/nan loss, spurious flat minima far
# away) — the failure mode seen with larger SIGMA. Must exceed the true displacement to be
# recovered (so comfortably above SIGMA's tail, ~2–3× SIGMA). None → no bound (raw BFGS).
MAX_DISPLACEMENT = None

# Multistart: BFGS often lands in a wrong, high-loss basin (the atom drifts to a spurious
# local min instead of the true position). Since we can only OBSERVE the loss (no ground
# truth), we retry from jittered starts and keep the run with the LOWEST final loss.
N_RESTARTS       = 5     # max BFGS runs per atom (0 or 1 = restarts off, a single run).
                         # Extra runs only fire when the current best is still above
                         # RESTART_LOSS_TOL.
RESTART_LOSS_TOL = 2.0   # a run below this is "good enough" → stop restarting. Also the
                         # loss ceiling for a recovery to count as "reliable" in STEP 6.
                         # UNITS FOLLOW `METRIC`: ppm of transport for "wasserstein",
                         # dimensionless for "rmse"/"cosine" — retune when you switch.
                         # Anchor it on the loss at the true geometry, printed by STEP 3.
RESTART_JITTER   = 0.4   # Å: σ of the Gaussian jitter that seeds each restart's start pos.

# Step 6 – statistics loop
RUN_STATS      = True  # set False to skip the multi-molecule statistics loop
N_MOL_STAT     = 100     # how many molecules to include (ignored if RUN_STATS=False)
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
    `CshNmrDataModule._to_graph`.  The neighbor list is rebuilt from scratch every
    call, which is required whenever positions change; since the Atoms carry a cell
    and pbc=True, `graph_from_ase` builds it with periodic images."""
    return functools.partial(
        gcnn.atomic.graph_from_ase,
        r_max=r_max,
        atom_include_keys=("numbers", "nmr_tensors"),
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


def min_image_distances(atoms, atom_k, pos_k=None, others_only=False):
    """Distances from atom k to every atom, under the minimum-image convention.

    CSH is a periodic solid, so a raw Euclidean distance is wrong near a cell face:
    an atom there can sit 0.5 Å from a neighbour's periodic image while looking many
    Å away in Cartesian coordinates. `mic=True` compares against the closest image
    (ASE falls back to plain distances when pbc is all-False, so this is also correct
    for a non-periodic molecule).

    :param pos_k: if given, atom k is first moved there (on a copy).
    :param others_only: drop atom k's own (zero) entry.
    """
    if pos_k is not None:
        atoms = atoms.copy()
        atoms.positions[atom_k] = pos_k
    idxs = np.arange(len(atoms))
    if others_only:
        idxs = np.delete(idxs, atom_k)
    return atoms.get_distances(atom_k, idxs, mic=bool(np.any(atoms.pbc)))


def perturb_atom(atoms, atom_k, sigma, min_dist, rng, max_attempts=200):
    """Displace atom k by Gaussian noise; reject if too close to any other atom
    (or to any of their periodic images — see `min_image_distances`).

    Returns (perturbed_atoms, perturbed_pos_k).
    Raises RuntimeError if no valid perturbation found after max_attempts.
    """
    pos_k_orig = atoms.positions[atom_k].copy()

    for _ in range(max_attempts):
        noise    = rng.normal(scale=sigma, size=3)
        new_pos  = pos_k_orig + noise
        atoms_new = atoms.copy()
        atoms_new.positions[atom_k] = new_pos
        if min_image_distances(atoms_new, atom_k, others_only=True).min() >= min_dist:
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
    """One fitted nucleus: which atoms it covers, its ppm axis, width and target curve.

    One of these per entry of `SPECTRUM_SPECIES`. They are built once from the target
    tensors and then held fixed for the whole optimisation, so neither the axis nor the
    target can move under the optimiser's feet.
    """

    symbol: str           # "Na", "Si", …
    weights: jnp.ndarray  # (n_atoms,) 1.0 on this species, 0 elsewhere
    grid: jnp.ndarray     # (GRID_POINTS,) ppm axis
    fwhm: float           # ppm, this nucleus' linewidth
    target: jnp.ndarray   # (GRID_POINTS,) target spectrum


def fwhm_for(symbol):
    """Linewidth of one nucleus. `FWHM` is either a single number or a per-species map."""
    if isinstance(FWHM, dict):
        if symbol not in FWHM:
            sys.exit(f"FWHM has no entry for {symbol!r}; add one or use a single float.")
        return float(FWHM[symbol])
    return float(FWHM)


def species_spectrum(tensors, spec: SpeciesSpectrum):
    """Spectrum of one nucleus from a full set of per-atom tensors.

    `spec.weights` selects the species, so the tensors of every other element are
    multiplied by zero rather than sliced out — slicing would break the traced shapes.
    """
    return nmr_spectra.mas_spectrum(
        tensors,
        spec.grid,
        fwhm=spec.fwhm,
        weights=spec.weights,
        lineshape=LINESHAPE,
        slope=REF_SLOPE,
        intercept=REF_INTERCEPT,
    )


def spectral_loss(tensors, specs):
    """Mean spectral mismatch over the fitted nuclei.

    Each nucleus is a separate experiment, so the LOSSES are averaged, not the spectra.
    Pure JAX, hence usable both inside the traced optimiser loss and on the host for the
    sanity checks and plots.
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

    Each nucleus gets its OWN grid, spanning its target peaks padded by `GRID_MARGIN`:
    ²³Na and ²⁹Si sit hundreds of ppm apart, so a shared axis would waste resolution on
    an empty region. Species the structure does not contain are skipped with a warning.
    """
    symbols = np.array(atoms.get_chemical_symbols()[:n_atoms])
    target_tensors = np.asarray(target_tensors)
    specs = []

    for symbol in species:
        mask = symbols == symbol
        if not mask.any():
            print(f"  ⚠ no {symbol} in this structure — skipping its spectrum")
            continue

        iso = np.trace(target_tensors[mask], axis1=1, axis2=2) / 3.0
        shifts = REF_SLOPE * iso + REF_INTERCEPT
        grid = nmr_spectra.make_grid(
            float(shifts.min()) - GRID_MARGIN,
            float(shifts.max()) + GRID_MARGIN,
            GRID_POINTS,
        )
        spec = SpeciesSpectrum(symbol, jnp.asarray(mask.astype(np.float32)), grid,
                               fwhm_for(symbol), None)
        specs.append(spec._replace(target=species_spectrum(jnp.asarray(target_tensors), spec)))

        print(f"  {symbol}: {int(mask.sum())} site(s), fwhm {spec.fwhm} ppm, "
              f"δ ∈ [{shifts.min():.1f}, {shifts.max():.1f}] ppm, "
              f"grid [{float(grid[0]):.1f}, {float(grid[-1]):.1f}]")

    if not specs:
        sys.exit(f"None of SPECTRUM_SPECIES={species} occurs in this structure.")
    return specs


def compute_loss(pred_tensors, specs):
    """Host-side spectral loss, for the STEP 3 sanity check and the prints.

    Identical formula to the traced loss the optimiser minimises, so the numbers in the
    log and in the plots are directly comparable.
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
    # Ignore non-finite losses (an L-BFGS line-search probe can land on a
    # degenerate geometry where the model returns inf/nan) when setting limits.
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
to_graph  = make_graph_builder(cfg.data.r_max)
predictor = make_predictor(module, params, to_graph)
print("  loaded.")

# ── STEP 0b: Reproduce (part of) the run's train/val/test split ────────────────
# reax.data.random_split draws from the Engine's default nnx.Rngs stream, which REAX
# always seeds as nnx.Rngs(0) unless a caller explicitly overrides it (nothing in this
# repo does). CshNmrDataModule.load_split_structures() replays that same (struct_type-
# grouped) split computation and returns the chosen partition as ASE Atoms — no saved
# seed or index list required, as long as the underlying data file (and `limit`) are
# unchanged. Atoms, not graphs: the whole demo MOVES an atom, so the neighbour list has
# to be rebuilt from the geometry at every step (see `make_graph_builder`).
if SPLIT not in ("train", "val", "test", "full"):
    sys.exit(f"SPLIT must be one of 'train'/'val'/'test'/'full', got {SPLIT!r}")

print(f"\nLoading the '{SPLIT}' split from {os.path.basename(RUN_DIR)}'s config …")
extra = dict(data_file=DATA_FILE)
if LIMIT is not None:
    extra["limit"] = LIMIT
datamodule = hydra.utils.instantiate(cfg.data, **extra, _convert_="object")

if SPLIT == "full":
    structures = datamodule._load_structures()   # everything, `limit`-restricted, unsplit
else:
    structures = datamodule.load_split_structures(SPLIT, limit=SPLIT_LIMIT)
print(f"  {len(structures)} structures in the '{SPLIT}' split.")

# ── STEP 1: Ground truth ───────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

if MOL_IDX is None:
    MOL_IDX = int(rng.integers(len(structures)))
    print(f"  MOL_IDX unset → picked structure {MOL_IDX} at random (SEED={SEED}).")

atoms_orig  = structures[MOL_IDX]        # ASE Atoms (original geometry, with cell/pbc)
graph_orig  = to_graph(atoms_orig)        # jraph.GraphsTuple (original)
n_atoms     = int(graph_orig.n_node[0])
positions   = np.array(graph_orig.nodes["positions"])   # (n_atoms, 3)

print(f"\n[{SPLIT}] structure {MOL_IDX}: {n_atoms} atoms")
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
print(f"\n  Building target spectra ({METRIC} loss, lineshape={LINESHAPE}):")
specs = build_species_spectra(atoms_orig, n_atoms, target_tensors, SPECTRUM_SPECIES)
n_sites = int(sum(float(sp.weights.sum()) for sp in specs))
print(f"  → fitting {len(specs)} spectrum/spectra over {n_sites} site(s); "
      f"atom k is {'' if any(float(sp.weights[atom_k]) > 0 for sp in specs) else 'NOT '}"
      f"among them")

# ── STEP 2: Perturbation ───────────────────────────────────────────────────────

print(f"\n[STEP 2] Perturbing atom {atom_k} with σ={SIGMA} Å …")
atoms_pert, pos_pert = perturb_atom(
    atoms_orig, atom_k, SIGMA, MIN_DIST, rng, MAX_PERTURB_ATTEMPTS
)
pos_true = positions[atom_k].copy()
displacement = np.linalg.norm(pos_pert - pos_true)
print(f"  |Δpos| = {displacement:.4f} Å")
print(f"  true pos:      {pos_true}")
print(f"  perturbed pos: {pos_pert}")

# ── STEP 3: Loss function + sanity check ───────────────────────────────────────

print(f"\n[STEP 3] Building loss function …")

# ── The supervisor's on-device minimiser (reproduced 1:1, for reference) ───────
# `gcnn.adapt(fun, wrt, outs=(what,))` wraps a graph function so it takes the
# optimised quantity as a positional arg (injected at `wrt`) and returns the
# scalar at `what`; `jax.scipy.optimize.minimize` then minimises it entirely on
# device. It is fast and gradient-based, but exposes no per-iteration trajectory,
# so the visualised runs below drive the SAME differentiable loss with scipy
# instead (see `minimize_atom`). Kept here as the canonical idiom for CSH scaling.
from collections.abc import Callable          # noqa: E402
from jax.scipy import optimize as jopt        # noqa: E402
import tensorial                              # noqa: E402


def minimize_fn(fun, what, wrt) -> Callable:
    graph_fn = gcnn.adapt(fun, wrt, outs=(what,))

    def minim(graph, x0, *, method, tol=None, options=None):
        # optimize() only takes 1D arrays, so flatten and un-flatten
        def to_minimize(value):
            value = value.reshape(x0.shape)
            return tensorial.as_array(graph_fn(graph, value)).flatten()[0]

        res = jopt.minimize(to_minimize, x0.flatten(), method=method, tol=tol, options=options)
        res = res._replace(x=res.x.reshape(x0.shape))
        return res

    return minim


# ── Differentiable NMR loss + gradient (the actual, observable optimiser) ──────

def _bound_disp(u, x0, max_disp, xp):
    """Map the unconstrained optimiser variable `u` to a position whose displacement
    from `x0` is smoothly capped at `max_disp`.

    Radial (isotropic) tanh squashing: with ``d = u − x0`` and ``r = |d|``, returns
    ``x0 + d · (R·tanh(r/R) / r)``, so the Euclidean displacement is
    ``|pos − x0| = R·tanh(r/R) < R`` for any `u`, and small steps (r ≪ R) are left
    essentially unchanged. Smooth everywhere (the +eps keeps r away from 0). `xp` is
    the array module (``jnp`` inside the traced loss, ``np`` on the host).
    ``max_disp=None`` → identity (raw, unbounded)."""
    if max_disp is None:
        return u
    d = u - x0
    r = xp.sqrt(xp.sum(d * d) + 1e-12)
    return x0 + d * (max_disp * xp.tanh(r / max_disp) / r)


def make_value_and_grad(module, params, atom_k, specs, x0=None, max_disp=None):
    """Build a JAX (loss, grad) function of atom k's optimiser variable.

    Given a graph (fixed topology) and the optimiser variable, it maps it to a
    (optionally displacement-bounded) position, injects it into the positions and
    runs the model — whose ``EdgeVectors`` layer RECOMPUTES the edge vectors from the
    positions, so reverse-mode autodiff flows all the way back. The tensors are then
    turned into spectra and compared with the targets, so the gradient chain is
    ``pos_k → tensors → spectra → loss`` and is exact for that topology. The predicted
    spectrum of the first fitted nucleus is carried as aux (for the GIF).
    """
    n_atoms = specs[0].weights.shape[0]
    x0j = None if x0 is None else jnp.asarray(x0)

    def _loss(graph, u):
        pos_k = _bound_disp(u, x0j, max_disp, jnp)
        pos  = graph.nodes["positions"].at[atom_k].set(pos_k)
        g    = gcnn.experimental.update_graph(graph).set(("nodes", "positions"), pos).get()
        out  = module._model.apply(params, g)
        pred = out.nodes["predicted_nmr_tensors"][:n_atoms]
        return spectral_loss(pred, specs), species_spectrum(pred, specs[0])

    return jax.jit(jax.value_and_grad(_loss, argnums=1, has_aux=True))


def make_loss_graph_fn(module, params, atom_k, specs,
                       recorder=None, x0=None, max_disp=None):
    """GraphFunction for the supervisor's ``minimize_fn``.

    It reads the optimiser variable from the custom field ``globals.pos_k`` (injected
    by ``gcnn.adapt``), maps it to a (optionally displacement-bounded) position, writes
    it into ``nodes.positions``, runs the model — whose ``EdgeVectors`` layer recomputes
    edge vectors from positions — synthesises one spectrum per fitted nucleus and
    stores their mean mismatch at ``globals.loss`` for ``minimize_fn`` to minimise.

    With `max_disp` set, the position is capped to ``|pos_k − x0| < max_disp`` via a
    smooth tanh reparametrisation (see `_bound_disp`), so BFGS can never drive the atom
    into far, degenerate geometries.

    If `recorder` is given, it is invoked on the host at EVERY evaluation (incl. BFGS
    line-search probes, not just accepted steps) with (pos_k, loss, spectrum) via
    ``jax.debug.callback(..., ordered=True)`` — the ACTUAL (bounded) pos_k, so the
    recorded trajectory reflects the real positions. That callback is transparent to
    autodiff, so jopt still differentiates the loss on device exactly as before.
    """
    n_atoms = specs[0].weights.shape[0]
    x0j = None if x0 is None else jnp.asarray(x0)

    def fun(graph):
        u     = graph.globals["pos_k"].reshape(3)
        pos_k = _bound_disp(u, x0j, max_disp, jnp)
        # positions come from graph_from_ase as a numpy array (graph0 is a closed-over
        # constant, not a jitted arg), so cast to jnp before the functional update.
        pos   = jnp.asarray(graph.nodes["positions"]).at[atom_k].set(pos_k)
        g     = gcnn.experimental.update_graph(graph).set(("nodes", "positions"), pos).get()
        out   = module._model.apply(params, g)
        pred  = out.nodes["predicted_nmr_tensors"][:n_atoms]
        loss  = spectral_loss(pred, specs)
        if recorder is not None:
            jax.debug.callback(recorder, pos_k, loss, species_spectrum(pred, specs[0]),
                               ordered=True)
        return gcnn.experimental.update_graph(out).set(("globals", "loss"),
                                                       loss.reshape(1)).get()

    return fun


_GRAD_CHECKED = False   # whether the gradient has already been finite-diff-verified


def _grad_check(vg, to_graph, atoms_base, atom_k, x0, eps=1e-3):
    """Finite-difference check of the JAX gradient at x0.

    Confirms that pos_k → loss is actually differentiable (i.e. the model
    recomputes edge vectors from positions). A near-zero JAX gradient or a large
    relative error means autodiff is NOT flowing back to pos_k.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(3)

    # Use ONE fixed graph (topology fixed at x0) for both the analytic gradient and
    # the finite differences, so they are comparable. (Rebuilding the neighbor list
    # per probe would let x±eps straddle a topology change and corrupt the FD.)
    a0 = atoms_base.copy()
    a0.positions[atom_k] = x
    graph0 = to_graph(a0)

    def loss_only(xv):
        (lv, _), _ = vg(graph0, jnp.asarray(xv, dtype=jnp.float32))
        return float(lv)

    (_, _), g_jax = vg(graph0, jnp.asarray(x, dtype=jnp.float32))
    g_jax = np.asarray(g_jax, dtype=np.float64)

    g_fd = np.array([(loss_only(x + eps * e) - loss_only(x - eps * e)) / (2 * eps)
                     for e in np.eye(3)])
    rel = np.linalg.norm(g_jax - g_fd) / (np.linalg.norm(g_fd) + 1e-12)
    print(f"    grad check @ x0:  |JAX|={np.linalg.norm(g_jax):.4f}  "
          f"|FD|={np.linalg.norm(g_fd):.4f}  rel.err={rel:.2e}")
    # A steep landscape + float32 model make the central-difference reference
    # itself noisy (truncation/round-off), so rel.err up to ~0.2 is expected and
    # benign. The real failure mode is a ~0 analytic gradient (autodiff not
    # reaching pos_k) or a gross mismatch.
    if np.linalg.norm(g_jax) < 1e-6:
        print("    ⚠ JAX gradient is ~0 — autodiff is NOT reaching pos_k (edges not "
              "recomputed?).")
    elif rel > 0.4:
        print(f"    ⚠ gradient mismatch (rel.err {rel:.2e}) — autodiff path suspect.")
    else:
        note = "  (float32 FD noise)" if rel > 0.1 else ""
        print(f"    gradient OK ✓{note}")


_RESTART_RNG = np.random.default_rng(SEED + 1)   # seeds the multistart jitter (reproducible)


def optimize_position(module, params, to_graph, atoms_base, atom_k,
                      specs, x0, pos_true):
    """Recover atom k's position with the supervisor's on-device minimiser
    (``minimize_fn``: ``gcnn.adapt`` + ``jax.scipy.optimize.minimize``, BFGS).

    atom k's position lives in a custom ``globals.pos_k`` field that the loss
    GraphFunction injects into ``nodes.positions`` before running the model, so we
    optimise exactly those 3 DOF, fully on device (fixed topology, built once at
    the start). The loss+gradient stay 100% jitted JAX on device; a host
    ``jax.debug.callback`` (AD-transparent) records EVERY evaluation into ``traj``
    and — if PRINT_EVALS — prints live progress, so the plots/GIF get the full path
    even though jopt itself exposes no per-iteration hook.

    Runs up to ``N_RESTARTS`` BFGS attempts from jittered starts and returns the one
    with the LOWEST final loss (multistart), stopping early once a run beats
    ``RESTART_LOSS_TOL``. Since we can only observe the loss (no ground truth), the
    loss is the selection criterion.

    Returns dict(result, traj, n_evals, n_runs).
    """
    global _GRAD_CHECKED
    x0_arr   = np.asarray(x0, dtype=np.float64).reshape(3)
    pos_true_arr = np.asarray(pos_true, dtype=np.float64).reshape(3)

    if GRAD_CHECK and not _GRAD_CHECKED:
        _GRAD_CHECKED = True
        vg = make_value_and_grad(module, params, atom_k, specs,
                                 x0=x0_arr, max_disp=MAX_DISPLACEMENT)
        _grad_check(vg, to_graph, atoms_base, atom_k, x0_arr)

    def _jittered_start():
        """A valid start = x0 + Gaussian jitter, kept away from other atoms (and
        from their periodic images — see `min_image_distances`)."""
        for _ in range(MAX_PERTURB_ATTEMPTS):
            cand = x0_arr + _RESTART_RNG.normal(scale=RESTART_JITTER, size=3)
            dists = min_image_distances(atoms_base, atom_k, pos_k=cand, others_only=True)
            if dists.min() >= MIN_DIST:
                return cand
        return x0_arr.copy()

    def _single_run(start, run_idx):
        """One BFGS minimisation from `start`; returns (result_with_pos_x, traj)."""
        # Fixed-topology graph at the start point, carrying the custom pos_k global.
        a0 = atoms_base.copy()
        a0.positions[atom_k] = start
        graph0 = to_graph(a0)
        graph0 = gcnn.experimental.update_graph(graph0).set(
            ("globals", "pos_k"), jnp.asarray(start, dtype=jnp.float32)).get()

        traj = dict(positions=[], loss=[], spectra=[])

        def _record(pos_k, loss, spectrum):
            pos_k = np.asarray(pos_k, dtype=np.float64)
            loss  = float(loss)
            traj["positions"].append(pos_k)
            traj["loss"].append(loss)
            traj["spectra"].append(np.asarray(spectrum, dtype=np.float64))
            if PRINT_EVALS:
                dpos = float(np.linalg.norm(pos_k - pos_true_arr))
                tag  = f"[run {run_idx}] " if N_RESTARTS > 1 else ""
                print(f"    {tag}eval {len(traj['loss']):4d}: "
                      f"loss={loss:9.4f}   |Δpos|={dpos:.4f} Å", flush=True)

        # The supervisor's minimiser applied to our spectral loss GraphFunction.
        fun    = make_loss_graph_fn(
            module, params, atom_k, specs,
            recorder=_record, x0=start, max_disp=MAX_DISPLACEMENT,
        )
        minim  = minimize_fn(fun, what="globals.loss", wrt="globals.pos_k")
        result = minim(graph0, jnp.asarray(start, dtype=jnp.float32),
                       method="BFGS", options=dict(maxiter=MAX_ITER))
        # BFGS optimises the unconstrained variable u; map it back to the real
        # (bounded) position and store it in result.x so downstream reads a position.
        pos_rec = _bound_disp(np.asarray(result.x, dtype=np.float64).reshape(3),
                              start, MAX_DISPLACEMENT, np)
        result  = result._replace(x=pos_rec)

        # …but do not trust that answer. `jax.scipy.optimize`'s BFGS can return an `x`
        # that is inconsistent with its own `fun` when the line search fails (status >= 2,
        # the normal outcome on this landscape): measured on one CSH run, the loss at the
        # returned x was 42.47 while res.fun claimed 4.31, and the returned point had
        # never been evaluated at all. The recorder captured EVERY evaluation, so take
        # the best point actually seen — BFGS evaluates each iterate, so nothing better
        # than this exists in the run. Selection is on the loss alone, which is all a
        # real deployment can observe.
        losses = np.asarray(traj["loss"], dtype=np.float64)
        if losses.size:
            finite = np.where(np.isfinite(losses), losses, np.inf)
            i_best = int(np.argmin(finite))
            if np.isfinite(finite[i_best]):
                result = result._replace(
                    x=np.asarray(traj["positions"][i_best], dtype=np.float64),
                    fun=finite[i_best],
                )
        return result, traj

    best = None
    total_evals = 0
    n_runs = 0
    # There is always at least one run: N_RESTARTS <= 1 just means "no restarts".
    for run_idx in range(max(1, N_RESTARTS)):
        start = x0_arr if run_idx == 0 else _jittered_start()
        result, traj = _single_run(start, run_idx)
        n_runs      += 1
        total_evals += int(result.nfev)

        lossf  = float(result.fun)
        rmsd_f = float(np.linalg.norm(np.asarray(result.x) - pos_true))
        loss0  = traj["loss"][0] if traj["loss"] else float("nan")
        rmsd0  = float(np.linalg.norm(start - pos_true))
        tag    = f"run {run_idx}: " if N_RESTARTS > 1 else f"{METHOD_LABEL}: "
        print(f"    {tag}{int(result.nfev)} evals  loss "
              f"{loss0:8.3f} → {lossf:8.3f}  |Δr| {rmsd0:.4f} → {rmsd_f:.4f} Å"
              f"  (status={int(result.status)}, success={bool(result.success)})")

        if best is None or lossf < best["loss_final"]:
            best = dict(result=result, traj=traj, loss_final=lossf)
        if lossf < RESTART_LOSS_TOL:
            break   # good enough — no need to restart

    if N_RESTARTS > 1:
        rmsd_b = float(np.linalg.norm(np.asarray(best["result"].x) - pos_true))
        print(f"    best of {n_runs} run(s): loss {best['loss_final']:.3f} ppm  "
              f"|Δr| {rmsd_b:.4f} Å", flush=True)

    return dict(result=best["result"], traj=best["traj"],
                n_evals=total_evals, n_runs=n_runs)


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
result = best["result"]
traj   = best["traj"]

pos_recovered = result.x.reshape(3)
rmsd          = float(np.linalg.norm(pos_recovered - pos_true))
# REAL success = the atom is back near its true position (positional |Δr| in Å),
# independent of any optimiser convergence flag.
recovered  = bool(rmsd < DIST_THRESHOLD)
n_evals    = best["n_evals"]
loss_init  = traj["loss"][0] if traj["loss"] else float("nan")
loss_final = float(result.fun)

print(f"    optimiser: {METHOD_LABEL}  ({n_evals} evals total)")
print(f"    loss: {loss_init:.6f} → {loss_final:.6f}")
print(f"    |Δr|(recovered, true) = {rmsd:.4f} Å  (started at {displacement:.4f} Å)")
print(f"    >>> RECOVERED: {recovered}  "
      f"(|Δr| {'<' if recovered else '≥'} {DIST_THRESHOLD} Å threshold)")
print(f"    recovered pos: {pos_recovered}")

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
# The curve tracks EVERY evaluation, line-search probes included; BFGS returns its last
# ACCEPTED iterate, which after a failed line search is not the last probe. Mark it, so
# the panel cannot be misread as ending at the reported result.
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
                    "n_evals", "loss_init", "loss_final", "dist_ang",
                    "recovered", "optimiser_stopped"])
        w.writerow([MOL_IDX, atom_k, f"{displacement:.6f}",
                    n_evals, f"{loss_init:.8f}",
                    f"{loss_final:.8f}", f"{rmsd:.6f}",
                    recovered, result.success])
    print(f"  Summary CSV saved to {CSV_SUMMARY_PATH}")

# ── GIF ───────────────────────────────────────────────────────────────────────
if SAVE_GIF:
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
    #   reliable = success AND loss_final < RESTART_LOSS_TOL   (observable-only proxy)
    stats     = dict(dist=[], loss=[], n_evals=[], n_runs=[], success=0, reliable=0, total=0)
    stat_meta = dict(mol=[], atom=[], species=[], dist0=[])   # one row per accepted case

    stat_start = MOL_IDX + 1
    stat_end   = min(stat_start + N_MOL_STAT, len(structures))
    for mol_i in range(stat_start, stat_end):
        atoms_i = structures[mol_i]
        graph_i = to_graph(atoms_i)
        ni      = int(graph_i.n_node[0])
        pos_i   = np.array(graph_i.nodes["positions"])
        dft_i   = np.array(graph_i.nodes["nmr_tensors"][:ni])

        for _ in range(N_ATOM_STAT):
            k_i = select_atom_k(atoms_i, ni, ATOM_K, rng_stat)
            if k_i is None:
                continue   # species not present in this molecule
            try:
                atoms_pert_i, pos_pert_i = perturb_atom(
                    atoms_i, k_i, SIGMA, MIN_DIST, rng_stat, MAX_PERTURB_ATTEMPTS
                )
            except RuntimeError:
                continue

            stat_meta["mol"].append(mol_i)
            stat_meta["atom"].append(k_i)
            stat_meta["species"].append(atoms_i.get_chemical_symbols()[k_i])
            stat_meta["dist0"].append(float(np.linalg.norm(pos_pert_i - pos_i[k_i])))

            # Target: DFT labels, or the model's OWN self-consistent prediction
            # at the true geometry.
            tgt_i = dft_i if LOSS_TARGET == "dft" else predictor(atoms_i)
            specs_i = build_species_spectra(atoms_i, ni, tgt_i, SPECTRUM_SPECIES)
            best_i = optimize_position(
                module, params, to_graph, atoms_i, k_i, specs_i,
                pos_pert_i.flatten(), pos_i[k_i],
            )
            dist_i = float(np.linalg.norm(best_i["result"].x.reshape(3) - pos_i[k_i]))
            loss_i = float(best_i["result"].fun)
            stats["dist"].append(dist_i)
            stats["loss"].append(loss_i)
            stats["n_evals"].append(best_i["n_evals"])
            stats["n_runs"].append(best_i["n_runs"])
            stats["total"] += 1
            ok_dist = dist_i < DIST_THRESHOLD
            ok_loss = loss_i < RESTART_LOSS_TOL
            if ok_dist:
                stats["success"] += 1
            if ok_dist and ok_loss:
                stats["reliable"] += 1

            flag = "✓ reliable" if (ok_dist and ok_loss) else ("~ close" if ok_dist else "✗ miss")
            print(f"  mol {mol_i:2d}  atom {k_i:2d} ({stat_meta['species'][-1]}) → "
                  f"|Δr|={dist_i:.3f} Å  loss={loss_i:.2f} ppm  "
                  f"(runs={best_i['n_runs']})  {flag}")

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
            f"(|Δr| < {DIST_THRESHOLD} Å AND loss < {RESTART_LOSS_TOL} ppm)"
        )
        print(
            f"  median |Δr| = {np.median(stats['dist']):.3f} Å  |  "
            f"mean |Δr| = {np.mean(stats['dist']):.3f} Å  |  "
            f"mean evals = {np.mean(stats['n_evals']):.0f}  |  "
            f"mean runs = {np.mean(stats['n_runs']):.1f}"
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
                        "dist_ang", "loss", "n_evals", "n_runs", "recovered", "reliable"])
            for i in range(len(stat_meta["mol"])):
                d = stats["dist"][i]
                l = stats["loss"][i]
                w.writerow([i, stat_meta["mol"][i], stat_meta["atom"][i],
                            stat_meta["species"][i], f"{stat_meta['dist0'][i]:.6f}",
                            f"{d:.6f}", f"{l:.8f}",
                            stats["n_evals"][i], stats["n_runs"][i],
                            bool(d < DIST_THRESHOLD),
                            bool(d < DIST_THRESHOLD and l < RESTART_LOSS_TOL)])
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
