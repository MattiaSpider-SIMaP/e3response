#!/usr/bin/env python3
"""
NMR-guided structure resolution demo — gradient-based, single (direct) model.

Perturbs the position of one atom in a QM9 molecule and recovers it by
minimising the RMSE between the predicted and target NMR shielding tensors,
using the EXACT gradient ∂loss/∂pos_k from JAX autodiff (the model recomputes
edge vectors from positions, so differentiation flows back to the atom).

Steps:
  1. Ground truth tensors (DFT labels or model prediction at true geometry).
  2. Gaussian perturbation of atom k with validity checks.
  3. Loss = RMSE(predicted, target) over atom k + neighbours.
  4. Gradient-based local minimisation (gcnn.adapt + jax.scipy.optimize BFGS, 3 DOF).
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

from e3response import keys
from e3response.data import qm9_nmr

logging.getLogger("reax").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Model checkpoint directory (must contain config.yaml and checkpoints/last.ckpt).
RUN_DIR = "/home/mattia/Desktop/ml_codes/nmr_diff_project/qm9_full_heavy/nequip_nmr"

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

# Perturbation
SIGMA                = 0.7  # Gaussian noise amplitude (Å)
MIN_DIST             = 0.2   # min allowed distance to any other atom (Å)
MAX_PERTURB_ATTEMPTS = 200   # retries for valid perturbation

# Loss function
# "dft"   → target = DFT label stored in the graph (node["nmr_tensors"])
# "model" → target = model prediction at the ORIGINAL (unperturbed) geometry
#            (tests self-consistency of the landscape, decoupled from DFT accuracy)
LOSS_TARGET     = "dft"
NEIGHBOR_CUTOFF = 5.0   # Å – neighbours of atom k included in the loss

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
MAX_DISPLACEMENT = 2.0

# Multistart: BFGS often lands in a wrong, high-loss basin (the atom drifts to a spurious
# local min instead of the true position). Since we can only OBSERVE the loss (no ground
# truth), we retry from jittered starts and keep the run with the LOWEST final loss.
N_RESTARTS       = 5     # max BFGS runs per atom (1 = restarts off). Extra runs only fire
                         # when the current best is still above RESTART_LOSS_TOL.
RESTART_LOSS_TOL = 5.0   # ppm: a run below this is "good enough" → stop restarting. Also the
                         # loss ceiling for a recovery to count as "reliable" in STEP 6.
                         # Keep near the model's own val RMSE; molecules whose achievable
                         # loss floor is higher just spend the budget and keep their best.
RESTART_JITTER   = 0.4   # Å: σ of the Gaussian jitter that seeds each restart's start pos.

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


def neighbor_mask(positions, atom_k, cutoff):
    """Boolean mask of atoms within cutoff of atom k (including k itself)."""
    dists = np.linalg.norm(positions - positions[atom_k], axis=1)
    return dists <= cutoff


def compute_loss(pred_tensors, target_tensors, mask):
    """RMSE between predicted and target NMR tensors over masked atoms (ppm).

    Used for the STEP 3 sanity check; the optimiser itself uses the JAX
    autodiff loss in ``make_value_and_grad``. RMSE keeps units consistent with
    the training metric val/nmr_tensors_rmse.
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

# Neighbours of k within NEIGHBOR_CUTOFF
nbr_mask = neighbor_mask(positions, atom_k, NEIGHBOR_CUTOFF)
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


def make_value_and_grad(module, params, atom_k, target_tensors, nbr_mask,
                        x0=None, max_disp=None):
    """Build a JAX (loss, grad) function of atom k's optimiser variable.

    Given a graph (fixed topology) and the optimiser variable, it maps it to a
    (optionally displacement-bounded) position, injects it into the positions and
    runs the model — whose ``EdgeVectors`` layer RECOMPUTES the edge vectors from the
    positions, so reverse-mode autodiff flows all the way back. The gradient is
    therefore exact for that topology. The predicted σ[k] is carried as aux (for the
    trajectory / ellipsoid GIF).
    """
    tgt     = jnp.asarray(target_tensors)             # (n_atoms, 3, 3)
    mask3   = jnp.asarray(nbr_mask)[:, None, None]     # (n_atoms, 1, 1)
    n_terms = float(np.sum(nbr_mask) * 9)
    x0j     = None if x0 is None else jnp.asarray(x0)

    def _loss(graph, u):
        pos_k = _bound_disp(u, x0j, max_disp, jnp)
        pos  = graph.nodes["positions"].at[atom_k].set(pos_k)
        g    = gcnn.experimental.update_graph(graph).set(("nodes", "positions"), pos).get()
        out  = module._model.apply(params, g)
        pred = out.nodes["predicted_nmr_tensors"][:tgt.shape[0]]
        diff = jnp.where(mask3, pred - tgt, 0.0)
        loss = jnp.sqrt(jnp.sum(diff ** 2) / n_terms)   # masked RMSE (ppm)
        return loss, pred[atom_k]

    return jax.jit(jax.value_and_grad(_loss, argnums=1, has_aux=True))


def make_loss_graph_fn(module, params, atom_k, target_tensors, nbr_mask,
                       recorder=None, x0=None, max_disp=None):
    """GraphFunction for the supervisor's ``minimize_fn``.

    It reads the optimiser variable from the custom field ``globals.pos_k`` (injected
    by ``gcnn.adapt``), maps it to a (optionally displacement-bounded) position, writes
    it into ``nodes.positions``, runs the model — whose ``EdgeVectors`` layer recomputes
    edge vectors from positions, so the whole thing is differentiable — and stores the
    masked NMR-tensor RMSE (ppm) at ``globals.loss`` for ``minimize_fn`` to minimise.

    With `max_disp` set, the position is capped to ``|pos_k − x0| < max_disp`` via a
    smooth tanh reparametrisation (see `_bound_disp`), so BFGS can never drive the atom
    into far, degenerate geometries.

    If `recorder` is given, it is invoked on the host at EVERY evaluation (incl. BFGS
    line-search probes, not just accepted steps) with (pos_k, loss, σ[k]) via
    ``jax.debug.callback(..., ordered=True)`` — the ACTUAL (bounded) pos_k, so the
    recorded trajectory reflects the real positions. That callback is transparent to
    autodiff, so jopt still differentiates the loss on device exactly as before.
    """
    tgt     = jnp.asarray(target_tensors)             # (n_atoms, 3, 3)
    mask3   = jnp.asarray(nbr_mask)[:, None, None]     # (n_atoms, 1, 1)
    n_terms = float(np.sum(nbr_mask) * 9)
    x0j     = None if x0 is None else jnp.asarray(x0)

    def fun(graph):
        u     = graph.globals["pos_k"].reshape(3)
        pos_k = _bound_disp(u, x0j, max_disp, jnp)
        # positions come from graph_from_ase as a numpy array (graph0 is a closed-over
        # constant, not a jitted arg), so cast to jnp before the functional update.
        pos   = jnp.asarray(graph.nodes["positions"]).at[atom_k].set(pos_k)
        g     = gcnn.experimental.update_graph(graph).set(("nodes", "positions"), pos).get()
        out   = module._model.apply(params, g)
        pred  = out.nodes["predicted_nmr_tensors"][:tgt.shape[0]]
        diff  = jnp.where(mask3, pred - tgt, 0.0)
        loss  = jnp.sqrt(jnp.sum(diff ** 2) / n_terms)
        if recorder is not None:
            jax.debug.callback(recorder, pos_k, loss, pred[atom_k], ordered=True)
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
                      target_tensors, nbr_mask, x0, pos_true):
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
    other_pos = np.delete(np.asarray(atoms_base.positions, dtype=np.float64), atom_k, axis=0)

    if GRAD_CHECK and not _GRAD_CHECKED:
        _GRAD_CHECKED = True
        vg = make_value_and_grad(module, params, atom_k, target_tensors, nbr_mask,
                                 x0=x0_arr, max_disp=MAX_DISPLACEMENT)
        _grad_check(vg, to_graph, atoms_base, atom_k, x0_arr)

    def _jittered_start():
        """A valid start = x0 + Gaussian jitter, kept away from other atoms."""
        for _ in range(MAX_PERTURB_ATTEMPTS):
            cand = x0_arr + _RESTART_RNG.normal(scale=RESTART_JITTER, size=3)
            if np.linalg.norm(other_pos - cand, axis=1).min() >= MIN_DIST:
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

        traj = dict(positions=[], loss=[], tensors=[])

        def _record(pos_k, loss, tensor_k):
            pos_k = np.asarray(pos_k, dtype=np.float64)
            loss  = float(loss)
            traj["positions"].append(pos_k)
            traj["loss"].append(loss)
            traj["tensors"].append(np.asarray(tensor_k, dtype=np.float64))
            if PRINT_EVALS:
                dpos = float(np.linalg.norm(pos_k - pos_true_arr))
                tag  = f"[run {run_idx}] " if N_RESTARTS > 1 else ""
                print(f"    {tag}eval {len(traj['loss']):4d}: "
                      f"loss={loss:9.4f} ppm   |Δpos|={dpos:.4f} Å", flush=True)

        # The supervisor's minimiser applied to our masked-RMSE loss GraphFunction.
        fun    = make_loss_graph_fn(
            module, params, atom_k, target_tensors, nbr_mask,
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
        return result, traj

    best = None
    total_evals = 0
    n_runs = 0
    for run_idx in range(N_RESTARTS):
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
                    recovered, result.success])
    print(f"  Summary CSV saved to {CSV_SUMMARY_PATH}")

# ── GIF ───────────────────────────────────────────────────────────────────────
if SAVE_GIF:
    print(f"  Saving GIF → {GIF_PATH} …")
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
    #   reliable = success AND loss_final < RESTART_LOSS_TOL   (observable-only proxy)
    stats     = dict(dist=[], loss=[], n_evals=[], n_runs=[], success=0, reliable=0, total=0)
    stat_meta = dict(mol=[], atom=[], species=[], dist0=[])   # one row per accepted case

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
                    atoms_i, k_i, SIGMA, MIN_DIST, rng_stat, MAX_PERTURB_ATTEMPTS
                )
            except RuntimeError:
                continue

            nbr_i = neighbor_mask(pos_i, k_i, NEIGHBOR_CUTOFF)
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
if SAVE_GIF:
    print(f"  GIF:     {GIF_PATH}")
if RUN_STATS:
    print(f"  Stats:   {STAT_PLOT_PATH}")
    if SAVE_CSV:
        print(f"           {STAT_CSV_PATH}")
