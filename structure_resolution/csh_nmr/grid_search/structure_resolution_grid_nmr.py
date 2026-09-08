#!/usr/bin/env python3
"""
NMR structure resolution for CSH by GRID SEARCH — the honest, deployment-shaped test.

The spectra variants perturb atom k by a small amount and try to walk it back, seeding
the search from a ball around the perturbed position. That measures how good the local
basin is, but it quietly leaks the answer: the starts already know roughly where the
atom is. A real refinement knows no such thing — it has a spectrum and a framework, and
must place the missing atom from scratch.

This script removes the leak. It takes the true structure only to BUILD the target
spectra, then forgets where atom k is and searches its position across the WHOLE unit
cell: a cloud of starts sampled over the cell in fractional coordinates, each run to a
local minimum by the same Adam descent as the QM9 spectral demo, all in parallel via
``vmap``. The converged positions are clustered by loss; the low-loss cluster(s) are the
answer. Whether any of them coincides with the true site is checked ONLY at the end, as
validation — the search itself never uses it.

Why CSH and not QM9: the cell is dense, so a grid over it lands every start among real
neighbours (a grid over the vacuum around a small molecule wastes most starts on
isolated-atom geometries the model cannot score). And "where do the Na sit in this cell
given the ²³Na spectrum?" is literally the target application.

Inherited from the QM9 spectral work and unchanged here:
- **Every nucleus is a separate experiment** — one spectrum and one loss term per entry
  of `SPECTRUM_SPECIES`, averaged, never summed. On CSH one Na site against one number is
  hopeless, so ²³Na is fitted together with ²⁹Si (every nucleus responds to the Na).
- **Adam, not BFGS.** `jax.scipy.optimize`'s strong-Wolfe line search returned
  status=3 (zoom failed) in 10/10 QM9 cases against the float32-noisy loss; Adam has no
  line search and converged in all of them, 4.6× faster.
- **Wasserstein metric** — point-by-point losses go flat once peaks stop overlapping.

CSH caveats: ²³Na is quadrupolar (spin 3/2); only the chemical-shift contribution is
synthesised here, not the second-order quadrupolar lineshape, so this is self-consistent
against DFT/model but not directly against a measured ²³Na spectrum without an EFG model.
The cell is periodic: neighbour lists and every distance use the minimum image.

Steps:
  1. Ground-truth spectra (from DFT labels or from the model at the true geometry).
  2. Grid of candidate positions across the cell (no perturbation, no positional prior).
  3. Loss = spectral mismatch, averaged over the fitted nuclei.
  4. Parallel Adam descent from every start (vmap), batched to bound memory.
  5. Cluster the converged positions by loss; validation-only check against the true site.
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
import numpy as np
import omegaconf
import reax
from tensorial import gcnn
from tensorial.gcnn import atomic

from e3response import keys, nmr_spectra, structure_search as ss, sr_viz

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
SPLIT_LIMIT = 10   # further restriction WITHIN the chosen SPLIT (ignored if SPLIT="full").

# Which structure and which atom to resolve.
MOL_IDX = 0      # index into the chosen split (not the raw dataset)
ATOM_K  = "Na"   # None → random atom; int → direct index; str → random atom of that species
SEED    = 42     # fixed seed for reproducibility

# Validity guard on the loss. The atom being placed can wander into a clash with the
# framework during descent, and the model stops being a function well before atoms touch.
# This fences off that region; a real refinement needs it too, so it is NOT benchmark-only.
# (The QM9 checkpoint blew up — NaN gradients — below ~0.45 Å; retune per checkpoint.)
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
# Defaults to "model" on CSH: the self-consistency test (loss@true ≡ 0) must pass BEFORE a
# DFT- or experiment-referenced result is worth interpreting, otherwise a failure cannot be
# attributed between the optimiser, the landscape and model error.
LOSS_TARGET = "dft"

# ── Spectrum ──────────────────────────────────────────────────────────────────
# Which nuclei are fitted. Each is an INDEPENDENT experiment: one spectrum and one loss
# term per species, averaged — never summed into a single curve. On CSH one Na site
# against a single σ_iso cannot pin 3 coordinates, so ²³Na is fitted WITH ²⁹Si: every
# nucleus responds to where the Na sits, so each extra species buys equations. Adding "O"
# is the most informative numerically and the least realistic physically (¹⁷O needs
# enrichment). Species absent from a structure are skipped with a warning.
SPECTRUM_SPECIES = ("Na",)
# Linewidth in ppm: a single float for all nuclei, or one value per species. ²³Na lines in
# CSH are broad (quadrupolar, unmodelled here); ²⁹Si MAS lines are narrower.
FWHM        = {"Na": 8.0, "Si": 3.0, "O": 8.0, "Ca": 8.0, "H": 2.0}
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
# The grid must stay wide enough to hold the peaks for the WHOLE optimiser excursion: if a
# line leaves the window, its area vanishes, normalisation blows up and both the loss and
# its gradient become artefacts. Na moves ~95 ppm/Å here, and grid search starts atoms
# anywhere in the cell, so keep this generous.
GRID_MARGIN = 150.0        # ppm of padding beyond the target's own peak range
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
ADAM_STEPS   = 500     # iterations per start
ADAM_LR0     = 0.08    # Å-scale step at the beginning (cosine-decayed to ADAM_LR1)
ADAM_LR1     = 0.004   # final step: sets the resolution the answer is polished to
PRINT_EVALS  = True   # per-start progress. With a vmapped multistart this is a firehose;
                       # the per-start summary printed at the end is usually what you want.

# Grid search. There is NO positional prior: the starts blanket the whole cell on a
# JITTERED regular grid — a lattice in fractional coordinates with a small random offset
# per point — mapped through the cell matrix, so every point sits inside the periodic
# structure among real neighbours. (A plain aligned grid would alias with the crystal
# symmetry; an i.i.d. random cloud clumps and leaves gaps. Jittering gets even coverage
# without aliasing.) The starts run in PARALLEL through `jax.vmap`; each relaxes locally
# (LOCAL_CAP) and the converged positions are clustered by loss.
#
# The coverage knob is a DENSITY, not a count: `START_SPACING` is the target spacing (Å)
# between grid points. A count would make the sampling density — and thus the per-start hit
# rate and the effective resolution — vary with cell size across the test set, confounding
# the statistics. Fixing the spacing keeps them constant, because a basin has a roughly
# constant size in Å³ (a property of the model, not of the cell). N_STARTS is DERIVED per
# structure from spacing and cell volume, and reported. Tune the spacing from the per-start
# hit rate STEP 6 measures, exactly as N_STARTS was tuned on QM9.
START_SPACING = 2   # Å — target spacing of the grid before clash rejection
MAX_STARTS    = 4096  # safety cap on the raw grid; a huge cell + tiny spacing is refused
# How many starts are in flight at once. `vmap` replicates the model's activations per lane
# and reverse-mode AD holds them for the backward pass, so memory grows with lanes × atoms;
# a bulk CSH cell has far more atoms than a QM9 molecule, so keep this modest. Every full
# batch shares one compilation, so batching costs nothing but bounds the footprint.
STARTS_PER_BATCH = 8
# Each start relaxes only within LOCAL_CAP of itself: u is mapped to
# pos_k = start + R·tanh((u − start)/R), so |pos_k − start| < R for any u. To tile the cell
# with no gaps this must be at least the grid spacing; too large and distant starts collapse
# onto the same minimum, wasting lanes. Set to LOCAL_CAP_FACTOR × START_SPACING at runtime.
LOCAL_CAP_FACTOR = 1.0
LOCAL_CAP = None      # derived in STEP 2 from START_SPACING; do not set by hand.

# Degeneracy detection — the primary deployment output. With no true position to check
# against, a confidently wrong answer is indistinguishable from a correct one from the loss
# alone. The grid search gives the natural signal: cluster the converged positions and, if
# several well-separated clusters share a near-equal low loss, the spectrum does not
# determine the site and it must be reported as unresolved rather than answered.
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
RUN_STATS      = False  # set False to skip the multi-molecule statistics loop
N_MOL_STAT     = 10     # how many molecules to include (ignored if RUN_STATS=False)
N_ATOM_STAT    = 1      # atoms per molecule
DIST_THRESHOLD = 0.2    # Å – "success" criterion

# Output
SAVE_CSV        = True   # write per-start + per-cluster CSVs
SAVE_GIF        = True    # animate the descent paths of the best starts (3-D)
SAVE_SPECTRA_CSV = True  # write the target vs best-cluster spectra as columns
GIF_TOP_N       = 8      # how many of the lowest-final-loss paths to animate
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
    # CSH has no dipole; keep only what CshNmrDataModule._to_graph carries. The Atoms
    # carry a cell and pbc=True, so graph_from_ase builds the neighbour list with images.
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


def fwhm_for(symbol):
    """Linewidth of one nucleus. `FWHM` is either a single number or a per-species map."""
    if isinstance(FWHM, dict):
        if symbol not in FWHM:
            sys.exit(f"FWHM has no entry for {symbol!r}; add one or use a single float.")
        return float(FWHM[symbol])
    return float(FWHM)


class SpeciesSpectrum(typing.NamedTuple):
    """One fitted nucleus: which atoms it covers, its ppm axis, linewidth and target curve.

    One of these per entry of `SPECTRUM_SPECIES`. They are built once from the target
    tensors and then held fixed for the whole optimisation, so the axis and the target
    never move under the optimiser's feet.
    """

    symbol: str          # "Na", "Si", …
    weights: jnp.ndarray  # (n_atoms,) 1.0 on this species, 0 elsewhere
    grid: jnp.ndarray     # (GRID_POINTS,) ppm axis
    fwhm: float           # linewidth for this nucleus (ppm)
    target: jnp.ndarray   # (GRID_POINTS,) target spectrum


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
        spec = SpeciesSpectrum(symbol, weights, grid, fwhm_for(symbol), None)
        target = species_spectrum(jnp.asarray(target_tensors), spec)
        specs.append(spec._replace(target=target))

        print(f"  {symbol}: {int(mask.sum())} site(s), fwhm {spec.fwhm} ppm, "
              f"δ ∈ [{shifts.min():.1f}, {shifts.max():.1f}] ppm, "
              f"grid [{float(grid[0]):.1f}, {float(grid[-1]):.1f}]")

    if not specs:
        sys.exit(f"None of SPECTRUM_SPECIES={species} occurs in this structure.")
    return specs


def compute_loss(pred_tensors, specs):
    """Host-side spectral loss, for the STEP 3 sanity check and the prints.

    Identical formula to the traced loss the optimiser minimises, so the numbers in
    the log and in the plots are directly comparable.
    """
    return float(spectral_loss(jnp.asarray(pred_tensors), specs))


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
# repo does). CshNmrDataModule.load_split_structures() replays that same (struct_type-
# grouped) split and returns the chosen partition as ASE Atoms carrying cell + pbc — no
# saved seed or index list required, as long as the data file (and `limit`) are unchanged.
# Atoms, not graphs: grid search MOVES an atom across the cell, so the neighbour list is
# rebuilt from the geometry at every step.
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

atoms_orig  = structures[MOL_IDX]              # ASE Atoms (original geometry, with cell/pbc)
graph_orig  = to_graph(atoms_orig)             # jraph.GraphsTuple (original)
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
print(f"\n  Building target spectra ({METRIC} loss, fwhm={FWHM} ppm, "
      f"lineshape={LINESHAPE}):")
specs = build_species_spectra(atoms_orig, n_atoms, target_tensors, SPECTRUM_SPECIES)
n_sites = int(sum(float(s.weights.sum()) for s in specs))
print(f"  → fitting {len(specs)} spectrum/spectra over {n_sites} site(s); "
      f"atom k is {'' if any(float(s.weights[atom_k]) > 0 for s in specs) else 'NOT '}"
      f"among them")

# ── STEP 2: The search cell (no perturbation, no positional prior) ─────────────
# Grid search does not move the atom away from a known truth — it forgets the truth
# entirely and blankets the cell with candidate positions. `pos_true` is kept ONLY to
# score the result at the very end; nothing in the search sees it.

CELL = np.asarray(atoms_orig.cell)                     # 3×3 row vectors (Å)
CELL_VOLUME = float(abs(np.linalg.det(CELL)))
pos_true = positions[atom_k].copy()

# LOCAL_CAP follows the density knob: each start relaxes within LOCAL_CAP_FACTOR × spacing,
# so the basins tile the cell with no gaps at any spacing. Set once, used for every
# structure (the spacing is fixed; only the derived start COUNT varies per cell).
LOCAL_CAP = float(LOCAL_CAP_FACTOR * START_SPACING)

_lengths = np.linalg.norm(CELL, axis=1)
_ns = np.maximum(1, np.round(_lengths / START_SPACING).astype(int))
print(f"\n[STEP 2] Grid search over the cell — no positional prior on atom {atom_k}.")
print(f"  cell volume = {CELL_VOLUME:.1f} Å³, spacing {START_SPACING} Å "
      f"→ {'×'.join(map(str, _ns))} = {int(np.prod(_ns))} raw grid points "
      f"(before clash rejection)")
print(f"  LOCAL_CAP = {LOCAL_CAP:.2f} Å ({LOCAL_CAP_FACTOR}× spacing)")

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


def _search_config():
    """Bundle this script's tuning constants into the shared engine's SearchConfig."""
    return ss.SearchConfig(
        adam_steps=ADAM_STEPS, adam_lr0=ADAM_LR0, adam_lr1=ADAM_LR1, local_cap=LOCAL_CAP,
        min_dist_guard=MIN_DIST_GUARD, guard_margin=GUARD_MARGIN, guard_weight=GUARD_WEIGHT,
        starts_per_batch=STARTS_PER_BATCH, dist_threshold=DIST_THRESHOLD,
        degeneracy_sep=DEGENERACY_SEP, degeneracy_tol=DEGENERACY_TOL, max_starts=MAX_STARTS)


def grid_search(module, params, to_graph, atoms_base, atom_k, specs, pos_true, cell):
    """Resolve atom k's position with NO prior: Adam descents from a jittered cell grid.

    Starts blanket the cell (`sample_grid_starts`) at density `START_SPACING`; each is
    relaxed within `LOCAL_CAP` of itself by the vmapped Adam (`make_multistart_adam`). The
    converged positions are clustered by loss (`cluster_minima`); the lowest-loss cluster
    centre is the answer. The number of starts is DERIVED from the spacing and this cell.

    `pos_true` is used ONLY to score the result afterwards — never in the search, which
    sees the loss alone, exactly as a real refinement would. Returns a dict with the best
    cluster, all clusters, the degeneracy alternatives, and validation-only diagnostics
    (distance of the best cluster to truth, and how many starts fell in the true basin).
    """
    pos_true_arr = np.asarray(pos_true, dtype=np.float64).reshape(3)
    cfg = _search_config()

    starts, n_raw = ss.sample_grid_starts(atoms_base, atom_k, START_SPACING, cell, cfg,
                                          _RESTART_RNG)
    n_starts = len(starts)   # derived: raw grid minus clash rejections
    void_frac = n_starts / max(n_raw, 1)

    # Size the neighbour capacity against geometries spread across the CELL, not just the
    # start geometry — otherwise the roaming atom overflows it silently and the per-candidate
    # loss is computed on an incomplete list (see make_neighbour_rebuilder / the topology
    # check in STEP 4). Build full geometries with atom k placed at a subsample of starts.
    rebuild = None
    if LIVE_NEIGHBOURS:
        sub = starts[np.random.default_rng(0).choice(
            n_starts, min(16, n_starts), replace=False)]
        probes = []
        for s in sub:
            p = np.asarray(atoms_base.positions, dtype=np.float64).copy()
            p[atom_k] = s
            probes.append(p)
        rebuild, _capacity = ss.make_neighbour_rebuilder(
            atoms_base, R_MAX, NEIGHBOUR_CAPACITY, probe_positions=probes)

    loss_fn = lambda pred: spectral_loss(pred, specs)     # ties the engine to the spectra
    out = ss.run_multistart(module, params, to_graph, atoms_base, atom_k, loss_fn, starts,
                            cfg, rebuild=rebuild)
    losses = out["losses"]                 # (n_starts,), inf where invalid
    positions_out = out["positions"]       # (n_starts, 3)
    all_losses = out["all_losses"]         # (n_starts, ADAM_STEPS) — kept for the GIF
    all_pos = out["all_pos"]               # (n_starts, ADAM_STEPS, 3)
    finite = losses
    n_valid = int(np.isfinite(finite).sum())
    if n_valid == 0:
        raise RuntimeError(
            "every start ended on an invalid geometry — check MIN_DIST_GUARD/START_SPACING."
        )

    clusters = ss.cluster_minima(finite, positions_out, cfg, cell)
    best = clusters[0]
    alternatives = ss.find_degenerate_alternatives(clusters, cfg, cell)

    # Validation only (needs the truth): distance of the reported answer, and how many
    # starts landed in the true basin — the per-start hit rate that sets the spacing budget.
    dist_to_true = ss.mic_distance(best["centre"], pos_true_arr, cell)
    hits = int(sum(ss.mic_distance(p, pos_true_arr, cell) < DIST_THRESHOLD
                   for p in positions_out))

    print(f"    {METHOD_LABEL}: {n_starts} starts ({n_raw} grid, {100*void_frac:.0f}% "
          f"clash-free) × {ADAM_STEPS} steps, {n_valid} valid → {len(clusters)} minima")
    print(f"    best cluster: loss {best['loss']:.3f}  ({best['size']} starts)  "
          f"|Δr to true| {dist_to_true:.4f} Å   "
          f"({hits}/{n_starts} starts in the true basin)", flush=True)
    if alternatives:
        alt_txt = ", ".join(f"{d:.2f} Å (loss {l:.3f})" for d, l in alternatives[:3])
        print(f"    → {len(alternatives)} further site(s) fit within "
              f"{100 * DEGENERACY_TOL:.0f}% of the best loss — {alt_txt}")
        print(f"      several candidate sites are spectrum-consistent (expected for a "
              f"periodic solid); report the map, not a single answer.")

    return dict(pos=best["centre"], loss=best["loss"], clusters=clusters,
                alternatives=alternatives, dist_to_true=dist_to_true, hits=hits,
                n_starts=n_starts, n_raw=n_raw, n_valid=n_valid,
                n_evals=n_starts * ADAM_STEPS,
                positions=positions_out, losses=finite,
                all_losses=all_losses, all_pos=all_pos, rebuild=rebuild)


# Sanity check — self-consistency of the target and that the site is not flat.
print("  Sanity check:")
pred_true    = predictor(atoms_orig)
loss_at_true = compute_loss(pred_true, specs)
print(f"    loss @ true position = {loss_at_true:.6f}")
# A random clash-free position in the cell should score MUCH worse than the truth; if it
# does not, atom k's spectrum barely depends on its position and no search can resolve it.
_prng = np.random.default_rng(SEED + 7)
_others_p = np.delete(np.asarray(atoms_orig.positions, dtype=np.float64), atom_k, axis=0)
for _ in range(1000):
    _probe = _prng.random(3) @ CELL
    if ss.min_image_min_distance(_probe, _others_p, CELL) >= MIN_DIST_GUARD + GUARD_MARGIN:
        break
_atoms_probe = atoms_orig.copy()
_atoms_probe.positions[atom_k] = _probe
l_probe = compute_loss(predictor(_atoms_probe), specs)
print(f"    loss @ a random cell position = {l_probe:.6f}")
if l_probe <= loss_at_true * 1.05:
    msg = (f"a random position ({l_probe:.4f}) scores no worse than the truth "
           f"({loss_at_true:.4f}) → flat landscape, resolution unlikely.")
    if LOSS_TARGET != "model":
        raise AssertionError(f"Sanity check FAILED: {msg}")
    print(f"    ⚠ Sanity check WEAK: {msg}")
else:
    print(f"    Sanity check PASSED ✓  (random pos is {l_probe - loss_at_true:+.4f} worse)")

# ── STEP 4: Grid search ────────────────────────────────────────────────────────

print(f"\n[STEP 4] Grid search ({METHOD_LABEL}) over the whole cell — no positional prior …")

best = grid_search(module, params, to_graph, atoms_orig, atom_k, specs, pos_true, CELL)

pos_recovered = np.asarray(best["pos"]).reshape(3)
rmsd          = best["dist_to_true"]
# The atom is "resolved" if the reported (lowest-loss) cluster sits at the true site.
# This uses the truth and so is a VALIDATION verdict, not something a real run could issue.
resolved   = bool(rmsd < DIST_THRESHOLD)
loss_final = float(best["loss"])

print(f"    {best['n_valid']}/{best['n_starts']} starts valid, "
      f"{len(best['clusters'])} distinct minima, {best['hits']}/{best['n_starts']} in the "
      f"true basin")
print(f"    loss @ best cluster = {loss_final:.6f}   (loss @ true = {loss_at_true:.6f})")
print(f"    |Δr|(best cluster, true) = {rmsd:.4f} Å")
print(f"    >>> RESOLVED: {resolved}  "
      f"(|Δr| {'<' if resolved else '≥'} {DIST_THRESHOLD} Å threshold)")
print(f"    recovered pos: {pos_recovered}")
# Benchmark-only triage, impossible in a real refinement: a wrong answer whose loss is at
# or below loss@true is a MODEL limit (its minimum is misplaced); one still above it is a
# SEARCH failure (more/denser starts would help).
if not resolved:
    if loss_final <= loss_at_true * (1.0 + DEGENERACY_TOL):
        print(f"    → model limit: loss {loss_final:.3f} ≤ {loss_at_true:.3f} at the true "
              f"geometry, so the model prefers the wrong site. A better checkpoint is the "
              f"only fix.")
    else:
        print(f"    → search failure: loss {loss_final:.3f} still above {loss_at_true:.3f} "
              f"at the true geometry. Denser sampling should help.")

# Topology self-check: recompute the best cluster's loss from a neighbour list rebuilt on
# the host, the ground truth for the topology. A large gap means the live list was wrong.
atoms_check = atoms_orig.copy()
atoms_check.positions[atom_k] = pos_recovered
loss_rebuilt = compute_loss(predictor(atoms_check), specs)
_gap = abs(loss_final - loss_rebuilt)
print(f"    topology check: search loss {loss_final:.6f}  vs  rebuilt-list "
      f"{loss_rebuilt:.6f}   (gap {_gap:.2e})")

# ── STEP 5: Evaluation and plots ──────────────────────────────────────────────

print(f"\n[STEP 5] Plotting …")

# One forward pass at the best cluster gives the recovered spectra for the figure, the CSV
# and the offline bundle. Everything the figures need is bundled to disk (plot_data.npz +
# plot_meta.json) so `replot_grid.py` can regenerate them later with no model and no GPU —
# tweak the aesthetics in e3response.sr_viz and re-run replot rather than the optimiser.
atoms_rec = atoms_orig.copy()
atoms_rec.positions[atom_k] = pos_recovered
pred_rec  = jnp.asarray(predictor(atoms_rec))

spectra_rows = []          # (symbol, grid, target, best_cluster) — used by the spectra CSV
_syms, _grids, _tgts, _best, _best_loss = [], [], [], [], []
for spec in specs:
    grid_np = np.asarray(spec.grid)
    y_tgt   = np.asarray(spec.target)
    y_rec   = np.asarray(species_spectrum(pred_rec, spec))
    l_rec_s = float(nmr_spectra.spectrum_loss(jnp.asarray(y_rec), spec.target,
                                              x=spec.grid, normalise_mode=NORMALISE,
                                              metric=METRIC))
    spectra_rows.append((spec.symbol, grid_np, y_tgt, y_rec))
    _syms.append(spec.symbol); _grids.append(grid_np); _tgts.append(y_tgt)
    _best.append(y_rec); _best_loss.append(l_rec_s)

clusters = best["clusters"]
bundle = dict(
    framework_pos=np.asarray(positions, dtype=float),
    framework_numbers=np.asarray(atoms_orig.get_atomic_numbers()[:n_atoms], dtype=int),
    cell=np.asarray(CELL, dtype=float) if CELL is not None else np.zeros((3, 3)),
    pos_true=np.asarray(pos_true, dtype=float),
    cand_pos=np.asarray(best["positions"], dtype=float),
    cand_loss=np.asarray(best["losses"], dtype=float),
    cluster_centres=(np.asarray([c["centre"] for c in clusters], dtype=float)
                     if clusters else np.zeros((0, 3))),
    cluster_losses=np.asarray([c["loss"] for c in clusters], dtype=float),
    cluster_sizes=np.asarray([c["size"] for c in clusters], dtype=int),
    all_pos=np.asarray(best["all_pos"], dtype=float),
    all_losses=np.asarray(best["all_losses"], dtype=float),
    spec_symbols=np.asarray(_syms),
    spec_grid=np.asarray(_grids, dtype=float),
    spec_target=np.asarray(_tgts, dtype=float),
    spec_best=np.asarray(_best, dtype=float),
    spec_best_loss=np.asarray(_best_loss, dtype=float),
    # ── metadata (→ plot_meta.json) ──
    atom_k=int(atom_k), metric=METRIC, normalise=NORMALISE, mol_idx=int(MOL_IDX),
    atom_symbol=atoms_orig.get_chemical_symbols()[atom_k], target_label=target_label,
    dist_threshold=float(DIST_THRESHOLD), degeneracy_tol=float(DEGENERACY_TOL),
    loss_final=float(loss_final), loss_at_true=float(loss_at_true), rmsd=float(rmsd),
    resolved=bool(resolved), n_valid=int(best["n_valid"]), n_starts=int(best["n_starts"]),
    hits=int(best["hits"]), gif_top_n=int(GIF_TOP_N), gif_max_frames=int(GIF_MAX_FRAMES),
    gif_fps=int(GIF_FPS), start_spacing=float(START_SPACING),
)
sr_viz.save_bundle(RESULTS_DIR, bundle)
print(f"  Plot bundle saved to {os.path.join(RESULTS_DIR, sr_viz.DATA_NAME)}")
sr_viz.grid_result_figure(bundle, PLOT_PATH)
print(f"  Plot saved to {PLOT_PATH}")

# ── CSV ───────────────────────────────────────────────────────────────────────
if SAVE_CSV:
    # One row per converged start: its final loss, position and distance to the truth.
    with open(CSV_TRAJ_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "loss", "pos_x", "pos_y", "pos_z", "dist_to_true_ang"])
        for i in range(len(best["positions"])):
            p = best["positions"][i]
            w.writerow([i, f"{best['losses'][i]:.8f}",
                        f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}",
                        f"{ss.mic_distance(p, pos_true, CELL):.6f}"])
    print(f"  Per-start CSV saved to {CSV_TRAJ_PATH}")

    # One row per distinct minimum (cluster).
    with open(CSV_SUMMARY_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "loss", "size", "pos_x", "pos_y", "pos_z",
                    "dist_to_true_ang", "is_best"])
        for r, c in enumerate(best["clusters"]):
            w.writerow([r, f"{c['loss']:.8f}", c["size"],
                        f"{c['centre'][0]:.6f}", f"{c['centre'][1]:.6f}",
                        f"{c['centre'][2]:.6f}",
                        f"{ss.mic_distance(c['centre'], pos_true, CELL):.6f}", r == 0])
    print(f"  Cluster CSV saved to {CSV_SUMMARY_PATH}")

if SAVE_SPECTRA_CSV:
    with open(CSV_SPECTRA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "delta_ppm", "target", "best_cluster"])
        for symbol, gx, y_t, y_r in spectra_rows:
            for i in range(len(gx)):
                w.writerow([symbol, f"{gx[i]:.4f}", f"{y_t[i]:.8e}", f"{y_r[i]:.8e}"])
    print(f"  Spectra CSV saved to {CSV_SPECTRA_PATH}")

# ── GIF: the descent paths of the best starts ──────────────────────────────────
# The GIF_TOP_N lowest-final-loss trajectories moving through the 3-D cell with fading
# trails coloured by final loss, alongside their loss curves. Drawn by e3response.sr_viz
# from the same bundle, so `replot_grid.py` can regenerate/retune it offline.
if SAVE_GIF:
    print(f"  Saving GIF → {GIF_PATH} …")
    sr_viz.grid_descent_gif(bundle, GIF_PATH, top_n=GIF_TOP_N,
                            max_frames=GIF_MAX_FRAMES, fps=GIF_FPS)
    print(f"    GIF saved.")

# ── STEP 6: Statistics (grid search over many structures) ─────────────────────

if not RUN_STATS:
    print("\n[STEP 6] Skipped (RUN_STATS=False).")
else:
    print("\n[STEP 6] Grid-search statistics over multiple structures …")
    print(f"  N_MOL_STAT={N_MOL_STAT}  N_ATOM_STAT={N_ATOM_STAT}  ATOM_K={ATOM_K!r}  "
          f"spacing={START_SPACING} Å  DIST_THRESHOLD={DIST_THRESHOLD} Å\n")

    #   resolved = |Δr(best cluster, true)| < DIST_THRESHOLD   (needs ground truth)
    #   reliable = resolved AND loss < RELIABLE_LOSS_TOL       (observable-only proxy)
    # loss_true and hits need ground truth and only serve to separate a SEARCH failure
    # (loss still above the one at the true site — denser sampling helps) from a MODEL limit
    # (loss at or below it — the model's minimum is misplaced). n_alt (rival clusters) is the
    # same warning from observables alone, the one that carries over to a real refinement.
    stats = dict(dist=[], loss=[], n_starts=[], hit_rate=[], resolved=0, reliable=0,
                 total=0, model_limited=0, search_failed=0)
    stat_meta = dict(mol=[], atom=[], species=[], loss_true=[], hits=[], n_alt=[],
                     n_clusters=[])

    stat_start = MOL_IDX + 1
    stat_end   = min(stat_start + N_MOL_STAT, len(structures))
    rng_stat   = np.random.default_rng(SEED + 100)
    for mol_i in range(stat_start, stat_end):
        atoms_i = structures[mol_i]
        graph_i = to_graph(atoms_i)
        ni      = int(graph_i.n_node[0])
        pos_i   = np.array(graph_i.nodes["positions"])
        dft_i   = np.array(graph_i.nodes["nmr_tensors"][:ni])
        cell_i  = np.asarray(atoms_i.cell)

        for _ in range(N_ATOM_STAT):
            k_i = select_atom_k(atoms_i, ni, ATOM_K, rng_stat)
            if k_i is None:
                continue   # species not present in this structure

            # Target spectra from the DFT labels or the model's own self-consistent
            # prediction at the true geometry. Rebuilt per structure (its own sites/range).
            tgt_i = dft_i if LOSS_TARGET == "dft" else predictor(atoms_i)
            try:
                specs_i = build_species_spectra(atoms_i, ni, tgt_i, SPECTRUM_SPECIES)
            except SystemExit:
                continue   # none of SPECTRUM_SPECIES in this structure
            loss_true_i = compute_loss(predictor(atoms_i), specs_i)

            res = grid_search(module, params, to_graph, atoms_i, k_i, specs_i,
                              pos_i[k_i], cell_i)
            dist_i = res["dist_to_true"]
            loss_i = float(res["loss"])
            rate_i = res["hits"] / max(res["n_starts"], 1)

            stat_meta["mol"].append(mol_i)
            stat_meta["atom"].append(k_i)
            stat_meta["species"].append(atoms_i.get_chemical_symbols()[k_i])
            stat_meta["loss_true"].append(loss_true_i)
            stat_meta["hits"].append(res["hits"])
            stat_meta["n_alt"].append(len(res["alternatives"]))
            stat_meta["n_clusters"].append(len(res["clusters"]))
            stats["dist"].append(dist_i)
            stats["loss"].append(loss_i)
            stats["n_starts"].append(res["n_starts"])
            stats["hit_rate"].append(rate_i)
            stats["total"] += 1

            ok_dist = dist_i < DIST_THRESHOLD
            ok_loss = loss_i < RELIABLE_LOSS_TOL
            if ok_dist:
                stats["resolved"] += 1
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
            print(f"  struct {mol_i:3d}  atom {k_i:3d} ({stat_meta['species'][-1]}) → "
                  f"|Δr|={dist_i:.3f} Å  loss={loss_i:.2f} (true {loss_true_i:.2f})  "
                  f"{res['hits']}/{res['n_starts']} on target, "
                  f"{len(res['clusters'])} minima, {len(res['alternatives'])} rivals  {flag}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    if stats["total"] == 0:
        print("  no samples")
    else:
        print(f"  resolved rate = {stats['resolved']}/{stats['total']} "
              f"(|Δr| < {DIST_THRESHOLD} Å)")
        print(f"  reliable rate = {stats['reliable']}/{stats['total']} "
              f"(|Δr| < {DIST_THRESHOLD} Å AND loss < {RELIABLE_LOSS_TOL})")
        print(f"  median |Δr| = {np.median(stats['dist']):.3f} Å  |  "
              f"mean |Δr| = {np.mean(stats['dist']):.3f} Å  |  "
              f"mean starts/structure = {np.mean(stats['n_starts']):.0f}")
        # Two failure modes, opposite responses — never report them as one number.
        print(f"  of the {stats['total'] - stats['resolved']} miss(es): "
              f"{stats['search_failed']} search failure(s) → lower START_SPACING;  "
              f"{stats['model_limited']} model limit(s) → the model's minimum is misplaced, "
              f"only a better checkpoint helps.")
        # The per-start hit rate sets the spacing budget: with mean rate r and n starts,
        # P(at least one hit) = 1 − (1 − r)^n. Averaged over structures for orientation.
        _rate = float(np.mean(stats["hit_rate"]))
        _nmean = float(np.mean(stats["n_starts"]))
        print(f"  mean per-start hit rate = {100 * _rate:.1f}%  → P(≥1 hit) at "
              f"{_nmean:.0f} starts ≈ {100 * (1 - (1 - _rate) ** _nmean):.0f}%")
        _flagged = int(np.sum(np.asarray(stat_meta["n_alt"]) > 0))
        print(f"  degeneracy warnings = {_flagged}/{stats['total']} case(s) had a rival "
              f"cluster within {100 * DEGENERACY_TOL:.0f}% of the best loss")

    # ── Per-case statistics CSV ────────────────────────────────────────────────
    if SAVE_CSV and stat_meta["mol"]:
        with open(STAT_CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "struct_idx", "atom_k", "species", "dist_ang", "loss",
                        "loss_at_true", "n_starts", "starts_on_target", "n_clusters",
                        "n_rival_clusters", "resolved", "reliable", "model_limited"])
            for i in range(len(stat_meta["mol"])):
                d = stats["dist"][i]; l = stats["loss"][i]; lt = stat_meta["loss_true"][i]
                ok = d < DIST_THRESHOLD
                w.writerow([i, stat_meta["mol"][i], stat_meta["atom"][i],
                            stat_meta["species"][i], f"{d:.6f}", f"{l:.8f}", f"{lt:.8f}",
                            stats["n_starts"][i], stat_meta["hits"][i],
                            stat_meta["n_clusters"][i], stat_meta["n_alt"][i],
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
if RUN_STATS and SAVE_CSV:
    print(f"  Stats:   {STAT_CSV_PATH}")
