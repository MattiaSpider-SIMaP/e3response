#!/usr/bin/env python3
"""
NMR-guided structure resolution demo — gradient-based, single (direct) model.

Perturbs the position of one atom in a csh structure and recovers it by
minimising the RMSE between the predicted and target NMR shielding tensors,
using the EXACT gradient ∂loss/∂pos_k from JAX autodiff (the model recomputes
edge vectors from positions, so differentiation flows back to the atom).

Optimiser: the shared `e3response.structure_search` engine — a vmapped Adam
multistart with the neighbour list rebuilt at every step (so the periodic
topology stays correct as the atom moves). This replaces the old
`jax.scipy.optimize` BFGS, whose strong-Wolfe line search failed on the
float32-noisy loss (status=3, never converging). The only experiment-specific
piece is `loss_fn(pred_tensors) -> scalar` = the masked NMR-tensor RMSE over
atom k + its neighbours.

Steps:
  1. Ground truth tensors (DFT labels or model prediction at true geometry).
  2. Gaussian perturbation of atom k with validity checks.
  3. Loss = RMSE(predicted, target) over atom k + neighbours.
  4. Adam multistart with a live neighbour list (structure_search, 3 DOF).
  5. Evaluation: |Δr|, loss curve, tensor ellipsoid; per-run GIF.
  6. (Optional) Statistics over many molecules.
"""

import csv
import datetime
import functools
import logging
import os
import sys

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

from e3response import keys, structure_search as ss

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
# "dft"   → target = DFT label stored in the graph (node["nmr_tensors"])
# "model" → target = model prediction at the ORIGINAL (unperturbed) geometry
#            (tests self-consistency of the landscape, decoupled from DFT accuracy)
LOSS_TARGET     = "dft"
NEIGHBOR_CUTOFF = 3.0   # Å – neighbours of atom k included in the loss

# Optimisation — the shared `structure_search` Adam multistart (see module docstring).
# `N_STARTS` starts are drawn in a ball around the perturbed position and descended in
# parallel by a vmapped Adam; each stays within `LOCAL_CAP` of its own start, so coverage
# comes from the spread of the starts. The neighbour list is rebuilt every step, so the
# periodic topology follows the atom as it moves.
METHOD_LABEL = "structure_search-Adam"
PRINT_EVALS  = True    # print the 10 best starts' loss + |Δr| after the run (not per step).

N_STARTS         = 64    # parallel Adam descents per atom (multistart width)
START_RADIUS     = 2.5   # Å: radius of the ball of starts around the perturbed position
STARTS_PER_BATCH = 4     # vmap lanes in flight at once (reverse-mode AD through the
                         # 71-atom periodic model is ~0.6 GB/lane; 8 lanes OOMs at ~4.7 GB)
ADAM_STEPS       = 500    # Adam iterations per start
ADAM_LR0         = 0.08   # initial Å-scale step (cosine-decayed to ADAM_LR1)
ADAM_LR1         = 0.004  # final step: the resolution the answer is polished to
# Each start relaxes within this Å of itself (smooth tanh cap), so a start can never fire
# the atom into a far, degenerate geometry. None → no bound.
LOCAL_CAP        = 1.5

# Clash guard: a smooth repulsion keeps atom k out of the model's blow-up zone.
MIN_DIST_GUARD   = 0.5   # Å below which the model blows up; fenced off by the penalty
GUARD_MARGIN     = 0.15  # Å above the guard where the penalty switches on
GUARD_WEIGHT     = 200.0 # penalty = weight · (guard + margin − d)²

# Live neighbour list: rebuild the graph's neighbours at every Adam step so the periodic
# topology stays correct as the atom moves (a frozen list goes stale — module docstring).
LIVE_NEIGHBOURS   = True
NEIGHBOUR_CAPACITY = None  # slots per atom, or None to estimate + verify against the geometry

# Observable degeneracy check (no ground truth): distinct minima whose loss rivals the best.
DEGENERACY_SEP   = 0.5   # Å — how far apart two minima must be to count as distinct
DEGENERACY_TOL   = 0.25  # relative loss gap below which a rival minimum is "as good"

# A recovery counts as "reliable" in STEP 6 when its final loss is below this (ppm) — an
# observable-only proxy, kept near the model's own validation RMSE.
RELIABLE_LOSS_TOL = 5.0

# Step 6 – statistics loop
RUN_STATS      = True  # set False to skip the multi-molecule statistics loop
N_MOL_STAT     = 100     # how many molecules to include (ignored if RUN_STATS=False)
N_ATOM_STAT    = 1      # atoms per molecule
DIST_THRESHOLD = 0.2    # Å – "success" criterion

# Output
SAVE_CSV        = True   # write trajectory + summary CSV files
SAVE_GIF        = True   # write one GIF per model (can be slow for many steps)
SAVE_TENSOR_GIF = False   # write tensor-ellipsoid GIF (predicted vs target at each step)
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
TENSOR_GIF_PATH      = os.path.join(RESULTS_DIR, "tensor_ellipsoid.gif")
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


def neighbor_mask(atoms, atom_k, cutoff):
    """Boolean mask of atoms within cutoff of atom k (including k itself).

    Minimum-image distances, so neighbours that only coordinate atom k across a cell
    boundary are still included in the loss."""
    return min_image_distances(atoms, atom_k) <= cutoff


def compute_loss(pred_tensors, target_tensors, mask):
    """RMSE between predicted and target NMR tensors over masked atoms (ppm).

    Used for the STEP 3 sanity check; the optimiser itself uses the JAX autodiff
    loss built by ``make_loss_fn`` (driven by ``structure_search``). RMSE keeps
    units consistent with the training metric val/nmr_tensors_rmse.
    """
    diff = pred_tensors[mask] - target_tensors[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def save_gif(traj, atoms_orig, positions, pos_true, pos_pert,
             n_atoms, atom_k, displacement, loss_at_true, target_tensor,
             gif_path, max_frames=120, fps=15):
    """Save an animated GIF of the optimisation trajectory.

    Layout (2×3): top row = 3-D trajectory, loss curve, distance curve;
    bottom-left = predicted-vs-target NMR tensor ellipsoid for atom k,
    animated in sync with the 3-D path.
    """
    pos_hist    = np.array(traj["positions"])   # (n_evals, 3)
    loss_curve  = np.array(traj["loss"])
    tensors_hist = np.array(traj["tensors"])    # (n_evals, 3, 3)
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

    # 2×2 layout: left column = 3-D path (top) + tensor ellipsoid (bottom);
    # right column = loss (top) + distance (bottom), stacked so there's no empty space.
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    gs  = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.30)
    ax3d    = fig.add_subplot(gs[0, 0], projection="3d")
    ax_ell  = fig.add_subplot(gs[1, 0], projection="3d")   # ellipsoid, under the 3-D path
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
    ax_loss.set_ylabel("RMSE loss (ppm)", fontsize=8)
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

    # tensor-ellipsoid panel (under the 3-D path): green = target (static),
    # model colour = predicted σ[k] at the current step (animated).
    sym_tgt    = (target_tensor + target_tensor.T) / 2.0
    eig_tgt, _ = np.linalg.eigh(sym_tgt)
    ell_norm   = float(np.abs(eig_tgt).max()) or 1.0
    # Forward-fill any non-finite predicted tensors (bad line-search probes) so the
    # ellipsoid animation never eigendecomposes a nan/inf matrix.
    tensors_hist = tensors_hist.copy()
    last_ok = target_tensor
    for _t in range(len(tensors_hist)):
        if np.all(np.isfinite(tensors_hist[_t])):
            last_ok = tensors_hist[_t]
        else:
            tensors_hist[_t] = last_ok
    xt, yt, zt = _ellipsoid_xyz(target_tensor, ell_norm)
    ax_ell.plot_surface(xt, yt, zt, alpha=0.20, color="limegreen", linewidth=0)
    _X0, _Y0, _Z0 = _ellipsoid_xyz(tensors_hist[frame_idxs[0]], ell_norm)
    pred_surf = [ax_ell.plot_surface(_X0, _Y0, _Z0, alpha=0.65, color=color, linewidth=0)]
    _elim = 1.3
    ax_ell.set_xlim(-_elim, _elim); ax_ell.set_ylim(-_elim, _elim); ax_ell.set_zlim(-_elim, _elim)
    ax_ell.set_title("σ[k] ellipsoid  (green=target, colour=pred)", fontsize=8)
    ax_ell.tick_params(labelsize=5)

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
        # redraw the predicted ellipsoid for this step
        pred_surf[0].remove()
        Xp, Yp, Zp = _ellipsoid_xyz(tensors_hist[i], ell_norm)
        pred_surf[0] = ax_ell.plot_surface(Xp, Yp, Zp, alpha=0.65, color=color, linewidth=0)
        return trail_line, cur_pt, loss_line, dist_line, step_txt, pred_surf[0]

    anim = FuncAnimation(fig, _update, frames=len(frame_idxs),
                         interval=1000 // fps, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _ellipsoid_xyz(tensor, norm):
    """Return (X, Y, Z) surface arrays for the tensor ellipsoid.

    Semi-axis lengths = |eigenvalues| / norm so that both target and predicted
    ellipsoids share the same scale (norm = max |eigenvalue| of the target).
    """
    sym = (tensor + tensor.T) / 2.0
    eig_vals, eig_vecs = np.linalg.eigh(sym)
    semi = np.abs(eig_vals) / norm
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)          # (30, 30, 3)
    ellipsoid = sphere @ (eig_vecs * semi).T        # rotate + scale
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def save_tensor_gif(traj, target_tensor, path, max_frames=120, fps=15):
    """Animate predicted vs target NMR shielding tensor as ellipsoids for atom k.

    Green (transparent) = target (fixed).
    Blue (opaque)       = predicted at the current optimisation step.
    Both are normalised by the largest absolute eigenvalue of the target so
    the green ellipsoid always fills the view and the blue one can be compared.
    """
    tensors = traj.get("tensors", [])
    if not tensors:
        print(f"  [tensor GIF] no tensors stored — skipping.")
        return

    n_steps = len(tensors)
    if n_steps > max_frames:
        frame_idxs = np.round(np.linspace(0, n_steps - 1, max_frames)).astype(int)
    else:
        frame_idxs = np.arange(n_steps)

    sym_tgt = (target_tensor + target_tensor.T) / 2.0
    eig_tgt, _ = np.linalg.eigh(sym_tgt)
    norm = float(np.abs(eig_tgt).max()) or 1.0

    xt, yt, zt = _ellipsoid_xyz(target_tensor, norm)

    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xt, yt, zt, alpha=0.20, color="limegreen", linewidth=0, label="target")

    X0, Y0, Z0 = _ellipsoid_xyz(tensors[int(frame_idxs[0])], norm)
    pred_surf = [ax.plot_surface(X0, Y0, Z0, alpha=0.65, color="steelblue", linewidth=0)]

    lim = 1.3
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim);  ax.set_zlim(-lim, lim)
    ax.set_xlabel("x", fontsize=7);  ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7);  ax.tick_params(labelsize=6)
    title_obj = ax.set_title("")

    def _update(fi):
        pred_surf[0].remove()
        step = int(frame_idxs[fi])
        Xp, Yp, Zp = _ellipsoid_xyz(tensors[step], norm)
        pred_surf[0] = ax.plot_surface(Xp, Yp, Zp, alpha=0.65,
                                        color="steelblue", linewidth=0)
        title_obj.set_text(
            f"NMR tensor ellipsoid  –  step {step + 1}/{n_steps}\n"
            f"loss={traj['loss'][step]:.4f}   green=target  blue=predicted"
        )
        return pred_surf

    anim = FuncAnimation(fig, _update, frames=len(frame_idxs),
                         interval=1000 // fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Tensor GIF → {path}")


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

# Neighbours of k within NEIGHBOR_CUTOFF
nbr_mask = neighbor_mask(atoms_orig, atom_k, NEIGHBOR_CUTOFF)
n_nbr    = int(nbr_mask.sum())
print(f"  neighbours within {NEIGHBOR_CUTOFF} Å: {n_nbr} atoms (including k)")

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

# ── Full-tensor loss, and the trajectory tensors for the GIF ───────────────────
# The optimiser itself is the shared `structure_search` engine (Adam multistart + live
# neighbour list); the ONLY experiment-specific piece it needs is `loss_fn(pred) -> scalar`.

def make_loss_fn(target_tensors, nbr_mask):
    """Return ``loss_fn(pred_tensors) -> scalar`` = masked NMR-tensor RMSE (ppm).

    RMSE of the predicted vs target shielding tensors over atom k + its neighbours
    (the `nbr_mask`), the same quantity the old BFGS loss minimised and the same units
    as the training metric ``val/nmr_tensors_rmse``. `pred_tensors` is the model's
    ``(n_atoms, 3, 3)`` output; the gradient chain ``pos_k → tensors → loss_fn`` is
    exact for the current topology (see `structure_search`)."""
    tgt     = jnp.asarray(target_tensors)              # (n_atoms, 3, 3)
    mask3   = jnp.asarray(nbr_mask)[:, None, None]      # (n_atoms, 1, 1)
    n_terms = float(np.sum(nbr_mask) * 9)

    def loss_fn(pred):
        diff = jnp.where(mask3, pred[:tgt.shape[0]] - tgt, 0.0)
        return jnp.sqrt(jnp.sum(diff ** 2) / n_terms)

    return loss_fn


def trajectory_tensors(module, params, atom_k, atoms_base, positions,
                       rebuild=None, chunk=16):
    """Predicted σ[k] tensor at each of `positions`, for the ellipsoid GIF.

    The vmapped Adam no longer records per-step tensors (with `N_STARTS` runs in flight
    that would be an (N_STARTS, ADAM_STEPS, 3, 3) array), so the GIF's ellipsoid is
    recomputed afterwards for the winning trajectory only — one forward pass per frame,
    batched in chunks of `chunk` (a ragged tail is padded up rather than recompiled)."""
    n_atoms = len(atoms_base)
    graph_base = to_graph(atoms_base)
    pos_all = jnp.asarray(atoms_base.positions, dtype=jnp.float32)

    @jax.jit
    @jax.vmap
    def _tensor(pos_k):
        pos = pos_all.at[atom_k].set(pos_k)
        g = gcnn.experimental.update_graph(graph_base).set(("nodes", "positions"), pos).get()
        if rebuild is not None:
            g = rebuild(g, pos)
        out = module._model.apply(params, g)
        return out.nodes["predicted_nmr_tensors"][atom_k]

    pos_arr = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    out = []
    for i in range(0, len(pos_arr), chunk):
        block = pos_arr[i:i + chunk]
        pad = chunk - len(block)
        if pad:
            block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
        res = np.asarray(_tensor(jnp.asarray(block)))
        out.append(res[:chunk - pad] if pad else res)
    return np.concatenate(out, axis=0)


def _search_config():
    """Build the `structure_search.SearchConfig` from this script's constants."""
    return ss.SearchConfig(
        adam_steps=ADAM_STEPS, adam_lr0=ADAM_LR0, adam_lr1=ADAM_LR1,
        local_cap=LOCAL_CAP if LOCAL_CAP is not None else 1e9,
        min_dist_guard=MIN_DIST_GUARD, guard_margin=GUARD_MARGIN, guard_weight=GUARD_WEIGHT,
        starts_per_batch=STARTS_PER_BATCH, dist_threshold=DIST_THRESHOLD,
        degeneracy_sep=DEGENERACY_SEP, degeneracy_tol=DEGENERACY_TOL,
    )


_RESTART_RNG = np.random.default_rng(SEED + 1)   # seeds the multistart starts (reproducible)


def optimize_position(module, params, to_graph, atoms_base, atom_k,
                      target_tensors, nbr_mask, x0, pos_true):
    """Recover atom k's position: `N_STARTS` parallel Adam descents (via the shared
    `structure_search` engine), best OBSERVED loss wins.

    `x0` is the prior on where the atom is — the perturbed position in this benchmark,
    a candidate site in a real refinement. `N_STARTS` starts are drawn in a ball of
    `START_RADIUS` around it and descended simultaneously by a vmapped Adam; each stays
    within `LOCAL_CAP` of its own start, and the neighbour list is rebuilt every step so
    the periodic topology follows the atom. Selection sees the LOSS alone (no ground
    truth); clustering / degeneracy use the minimum-image convention (periodic cell).

    `pos_true` is used ONLY for printing and the returned diagnostics; it never enters
    the selection. Returns dict(pos, loss, traj, loss_at_guess, n_evals, n_starts,
    n_valid, alternatives, hits, rebuild, success).
    """
    x0_arr = np.asarray(x0, dtype=np.float64).reshape(3)
    pos_true_arr = np.asarray(pos_true, dtype=np.float64).reshape(3)
    other_pos = np.delete(np.asarray(atoms_base.positions, dtype=np.float64), atom_k, axis=0)
    cell = np.asarray(atoms_base.cell) if bool(np.any(atoms_base.pbc)) else None
    cfg_s = _search_config()
    loss_fn = make_loss_fn(target_tensors, nbr_mask)

    # The prior itself is always one of the starts, so the multistart can never do worse
    # than a single run from the guess.
    starts = ss.sample_ball_starts(x0_arr, other_pos, N_STARTS, START_RADIUS, cfg_s,
                                   _RESTART_RNG)
    starts[0] = x0_arr

    # One rebuilder per structure (captures the cell/pbc and the slot capacity, fixed for
    # the run). Size the capacity against a spread of the starts so it covers the roam.
    rebuild = None
    if LIVE_NEIGHBOURS:
        probes = [np.where(np.arange(len(atoms_base))[:, None] == atom_k, s, atoms_base.positions)
                  for s in starts[:min(16, len(starts))]]
        rebuild, _capacity = ss.make_neighbour_rebuilder(
            atoms_base, float(cfg.data.r_max), NEIGHBOUR_CAPACITY, probe_positions=probes)

    out = ss.run_multistart(module, params, to_graph, atoms_base, atom_k, loss_fn, starts,
                            cfg_s, rebuild=rebuild)
    losses = out["losses"]
    positions_out = out["positions"]
    all_losses = out["all_losses"]
    all_pos = out["all_pos"]

    finite = np.where(np.isfinite(losses), losses, np.inf)
    n_valid = int(np.isfinite(finite).sum())
    if n_valid == 0:
        raise RuntimeError(
            "every start ended on an invalid geometry — check MIN_DIST_GUARD/START_RADIUS."
        )
    i_best = int(np.argmin(finite))

    # Trajectory of the winning start, for the plots and the GIF (tensors filled on demand).
    traj = dict(
        positions=[np.asarray(p, dtype=np.float64) for p in np.asarray(all_pos)[i_best]],
        loss=[float(v) for v in np.asarray(all_losses)[i_best]],
        tensors=[],
    )

    # Observable degeneracy check (min-image), plus a ground-truth diagnostic kept apart.
    clusters = ss.cluster_minima(finite, positions_out, cfg_s, cell=cell)
    alternatives = ss.find_degenerate_alternatives(clusters, cfg_s, cell=cell)
    hits = int((np.linalg.norm(positions_out - pos_true_arr, axis=1) < DIST_THRESHOLD).sum())
    loss_at_guess = float(all_losses[0][0])   # starts[0] is the prior guess itself

    if PRINT_EVALS:
        order = np.argsort(finite)[:10]
        for rank, i in enumerate(order):
            print(f"      start {int(i):3d} (rank {rank}): loss {finite[i]:8.3f}  "
                  f"|Δr| {np.linalg.norm(positions_out[i] - pos_true_arr):.4f} Å")

    rmsd_b = float(np.linalg.norm(positions_out[i_best] - pos_true_arr))
    rmsd0 = float(np.linalg.norm(x0_arr - pos_true_arr))
    print(f"    {METHOD_LABEL}: {N_STARTS} starts × {ADAM_STEPS} steps  "
          f"loss {loss_at_guess:8.3f} → {finite[i_best]:8.3f} ppm  "
          f"|Δr| {rmsd0:.4f} → {rmsd_b:.4f} Å  "
          f"({hits}/{N_STARTS} starts reached the true basin)", flush=True)
    if alternatives:
        alt_txt = ", ".join(f"{d:.2f} Å (loss {l:.3f})" for d, l in alternatives[:3])
        print(f"    ⚠ DEGENERATE: {len(alternatives)} distinct position(s) fit within "
              f"{100 * DEGENERACY_TOL:.0f}% of the best loss — {alt_txt}")
        print(f"      the tensors do not determine this site; treat the answer as "
              f"unresolved rather than correct.")

    return dict(pos=positions_out[i_best], loss=float(finite[i_best]), traj=traj,
                loss_at_guess=loss_at_guess, n_evals=N_STARTS * ADAM_STEPS,
                n_starts=N_STARTS, n_valid=n_valid, alternatives=alternatives,
                hits=hits, rebuild=rebuild, success=bool(finite[i_best] < RELIABLE_LOSS_TOL))


# Sanity check
print("  Sanity check:")
pred_true = predictor(atoms_orig)
pred_pert = predictor(atoms_pert)
loss_at_true = compute_loss(pred_true, target_tensors, nbr_mask)
l_pert       = compute_loss(pred_pert, target_tensors, nbr_mask)
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
    module, params, to_graph, atoms_orig, atom_k, target_tensors, nbr_mask, x0, pos_true,
)
traj   = best["traj"]

pos_recovered = np.asarray(best["pos"]).reshape(3)
rmsd          = float(np.linalg.norm(pos_recovered - pos_true))
# REAL success = the atom is back near its true position (positional |Δr| in Å),
# independent of any optimiser convergence flag.
recovered  = bool(rmsd < DIST_THRESHOLD)
n_evals    = best["n_evals"]
loss_init  = best["loss_at_guess"]
loss_final = float(best["loss"])

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

# Layout: loss | distance | NMR shift comparison.
fig = plt.figure(figsize=(15, 4), facecolor="white")
gs  = fig.add_gridspec(1, 3, hspace=0.45, wspace=0.35)

loss_curve = np.array(traj["loss"])

# Column 0: loss vs optimisation step
ax_loss = fig.add_subplot(gs[0, 0])
ax_loss.semilogy(np.arange(len(loss_curve)), loss_curve, color=MODEL_COLOR, linewidth=1.5)
ax_loss.axhline(loss_at_true, color="limegreen", linestyle="--",
                linewidth=1.0, label="loss @ true pos")
ax_loss.set_xlabel("Evaluation #", fontsize=8)
ax_loss.set_ylabel("RMSE loss (ppm)", fontsize=8)
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
ax_dist.set_xlabel("Evaluation #", fontsize=8)
ax_dist.set_ylabel("|pos_k − pos_true| (Å)", fontsize=8)
ax_dist.set_title("Distance to true pos", fontsize=9)
ax_dist.legend(fontsize=7)
ax_dist.tick_params(labelsize=7)

# Column 2: NMR isotropic shift comparison over neighbour atoms
ax_nmr   = fig.add_subplot(gs[0, 2])
nbr_idxs = np.where(nbr_mask)[0]

atoms_rec = atoms_orig.copy()
atoms_rec.positions[atom_k] = pos_recovered
pred_rec  = predictor(atoms_rec)
pred_pert = predictor(atoms_pert)
pred_tgt  = predictor(atoms_orig) if LOSS_TARGET == "model" else target_tensors

iso_tgt  = np.array([pred_tgt[i].trace() / 3 for i in nbr_idxs])
iso_pert = np.array([pred_pert[i].trace() / 3 for i in nbr_idxs])
iso_rec  = np.array([pred_rec[i].trace() / 3 for i in nbr_idxs])

x_pos = np.arange(len(nbr_idxs))
bw    = 0.25
ax_nmr.bar(x_pos - bw, iso_tgt,  width=bw, label="target",    color="limegreen", alpha=0.8)
ax_nmr.bar(x_pos,      iso_pert, width=bw, label="perturbed", color="orangered", alpha=0.8)
ax_nmr.bar(x_pos + bw, iso_rec,  width=bw, label="recovered", color=MODEL_COLOR, alpha=0.8)
ax_nmr.set_xticks(x_pos)
ax_nmr.set_xticklabels(
    [f"{atoms_orig.get_chemical_symbols()[i]}[{i}]" for i in nbr_idxs],
    fontsize=7, rotation=45, ha="right",
)
ax_nmr.set_ylabel("σ isotropic (ppm)", fontsize=8)
ax_nmr.set_title("NMR shift comparison", fontsize=9)
ax_nmr.legend(fontsize=7)
ax_nmr.tick_params(labelsize=7)

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
                    recovered, best["success"]])
    print(f"  Summary CSV saved to {CSV_SUMMARY_PATH}")

# ── GIF ───────────────────────────────────────────────────────────────────────
if SAVE_GIF:
    print(f"  Saving GIF → {GIF_PATH} …")
    # The vmapped Adam does not record per-step σ[k]; recompute it for the winning
    # trajectory only (one forward pass per frame) so the ellipsoid panel animates.
    traj["tensors"] = list(trajectory_tensors(
        module, params, atom_k, atoms_orig, traj["positions"], rebuild=best["rebuild"]))
    save_gif(
        traj, atoms_orig, positions, pos_true, pos_pert,
        n_atoms, atom_k, displacement,
        loss_at_true, target_tensors[atom_k],
        GIF_PATH, max_frames=GIF_MAX_FRAMES, fps=GIF_FPS,
    )
    print(f"    GIF saved.")

if SAVE_TENSOR_GIF:
    print(f"  Saving tensor GIF → {TENSOR_GIF_PATH} …")
    save_tensor_gif(
        traj, target_tensors[atom_k],
        TENSOR_GIF_PATH, max_frames=GIF_MAX_FRAMES, fps=GIF_FPS,
    )

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
    #   reliable = success AND loss_final < RELIABLE_LOSS_TOL  (observable-only proxy)
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

            nbr_i = neighbor_mask(atoms_i, k_i, NEIGHBOR_CUTOFF)
            stat_meta["mol"].append(mol_i)
            stat_meta["atom"].append(k_i)
            stat_meta["species"].append(atoms_i.get_chemical_symbols()[k_i])
            stat_meta["dist0"].append(float(np.linalg.norm(pos_pert_i - pos_i[k_i])))

            # Target: DFT labels, or the model's OWN self-consistent prediction
            # at the true geometry.
            tgt_i = dft_i if LOSS_TARGET == "dft" else predictor(atoms_i)
            best_i = optimize_position(
                module, params, to_graph, atoms_i, k_i, tgt_i, nbr_i,
                pos_pert_i.flatten(), pos_i[k_i],
            )
            dist_i = float(np.linalg.norm(np.asarray(best_i["pos"]).reshape(3) - pos_i[k_i]))
            loss_i = float(best_i["loss"])
            stats["dist"].append(dist_i)
            stats["loss"].append(loss_i)
            stats["n_evals"].append(best_i["n_evals"])
            stats["n_runs"].append(best_i["n_starts"])
            stats["total"] += 1
            ok_dist = dist_i < DIST_THRESHOLD
            ok_loss = loss_i < RELIABLE_LOSS_TOL
            if ok_dist:
                stats["success"] += 1
            if ok_dist and ok_loss:
                stats["reliable"] += 1

            flag = "✓ reliable" if (ok_dist and ok_loss) else ("~ close" if ok_dist else "✗ miss")
            print(f"  mol {mol_i:2d}  atom {k_i:2d} ({stat_meta['species'][-1]}) → "
                  f"|Δr|={dist_i:.3f} Å  loss={loss_i:.2f} ppm  "
                  f"(starts={best_i['n_starts']})  {flag}")

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
            f"mean evals = {np.mean(stats['n_evals']):.0f}  |  "
            f"mean starts = {np.mean(stats['n_runs']):.1f}"
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
                            bool(d < DIST_THRESHOLD and l < RELIABLE_LOSS_TOL)])
        print(f"  Statistics CSV saved to {STAT_CSV_PATH}")

print(f"\nDone.")
print(f"  Plot:    {PLOT_PATH}")
if SAVE_CSV:
    print(f"  CSV:     {CSV_TRAJ_PATH}")
    print(f"           {CSV_SUMMARY_PATH}")
if SAVE_GIF:
    print(f"  GIF:     {GIF_PATH}")
if RUN_STATS:
    print(f"  Stats:   {STAT_PLOT_PATH}")
    if SAVE_CSV:
        print(f"           {STAT_CSV_PATH}")
