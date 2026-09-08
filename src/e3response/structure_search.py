"""Gradient structure-resolution search engine, shared by the demo scripts.

This is the loss-agnostic core extracted from the NMR structure-resolution scripts. It
places one atom by minimising an arbitrary differentiable loss on the model's prediction,
using Adam with random restarts and a neighbour list rebuilt at every step. The only thing
a caller supplies that ties it to a particular experiment is ``loss_fn(pred_tensors) ->
scalar`` — a spectral mismatch for the spectra scripts, a per-atom tensor RMSE for the
full-tensor scripts. Everything else (the optimiser, the live neighbour list, the guard
against clashes, the start sampling and the clustering) is identical across them and lives
here, so a fix lands once rather than in every script.

Why Adam and not ``jax.scipy.optimize`` BFGS: benchmarked on 10 QM9 molecules the BFGS
line search failed (``status=3``, zoom failed) in 10 cases out of 10 on the float32-noisy
loss, never converging. Adam has no line search and beat it on the final loss in every
case, 4.6× faster.

Why a live neighbour list: the graph is built once and only ``nodes.positions`` changes
inside the jitted loop, so a frozen list goes stale as the atom moves (measured on CSH:
loss 4.33 vs 0.60 after 0.557 Å). `make_neighbour_rebuilder` recomputes it with a
fixed-shape search, so it stays exact at every step with no recompilation.

Two correctness details that are easy to get wrong and were expensive to find:

- The clash distance MUST be computed from the frozen framework positions, never through
  the neighbour finder — differentiating through the finder hits ``sqrt(0)`` on the
  self-distance diagonal and returns a NaN gradient, which poisons Adam's moments and
  kills the lane for the rest of the run.
- Both the model gradient AND the penalty gradient are scrubbed with ``nan_to_num`` before
  they touch Adam's state; and the invalid-geometry sentinel is ``inf`` restored with
  ``jnp.where(isnan, inf, .)`` rather than ``nan_to_num`` (whose ``posinf`` default rewrites
  inf to 3.4e38, a finite value that then slips past every ``isfinite`` check downstream).
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tensorial import gcnn
from tensorial.geometry import jax_neighbours

from e3response import keys

Array = jax.Array


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    """Everything the search needs that a script might tune. Passed explicitly so nothing
    reads module globals — the scripts set these differently (QM9 vs CSH, spectra vs
    tensors)."""

    adam_steps: int = 500        # Adam iterations per start
    adam_lr0: float = 0.08       # Å-scale step at the start (cosine-decayed to adam_lr1)
    adam_lr1: float = 0.004      # final step: the resolution the answer is polished to
    local_cap: float = 1.5       # each start relaxes within this Å of itself (tanh cap)
    min_dist_guard: float = 0.5  # Å below which the model blows up; fenced off by a penalty
    guard_margin: float = 0.15   # Å above the guard where the penalty switches on
    guard_weight: float = 200.0  # penalty = weight · (guard + margin − d)²
    starts_per_batch: int = 16   # vmap lanes in flight at once (bounds GPU memory)
    dist_threshold: float = 0.2  # Å — validation success criterion (needs ground truth)
    degeneracy_sep: float = 0.5  # Å — how far apart two minima must be to count as distinct
    degeneracy_tol: float = 0.25 # relative loss gap below which a rival minimum is "as good"
    max_starts: int = 4096       # safety cap on a derived grid


# ── geometry helpers ───────────────────────────────────────────────────────────

def bound_disp(u, x0, max_disp, xp=jnp):
    """Map the unconstrained variable ``u`` to a position capped at ``max_disp`` from ``x0``.

    Radial tanh squashing: with ``d = u − x0`` and ``r = |d|``, returns
    ``x0 + d · (R·tanh(r/R)/r)``, so ``|pos − x0| = R·tanh(r/R) < R`` for any ``u`` and small
    steps are essentially unchanged. ``xp`` is the array module (``jnp`` traced, ``np`` host).
    ``max_disp=None`` → identity."""
    if max_disp is None:
        return u
    d = u - x0
    r = xp.sqrt(xp.sum(d * d) + 1e-12)
    return x0 + d * (max_disp * xp.tanh(r / max_disp) / r)


def min_distance_to_others(graph, positions, atom_k):
    """Traced distance from atom k to its closest other atom, minimum-image when periodic.

    Non-finite positions propagate NaN, which fails every comparison — exactly what the
    validity guard wants."""
    delta = positions - positions[atom_k]
    if keys.CELL in graph.globals:
        cell = jnp.asarray(graph.globals[keys.CELL]).reshape(3, 3)
        frac = delta @ jnp.linalg.inv(cell)
        delta = (frac - jnp.round(frac)) @ cell
    dist = jnp.linalg.norm(delta, axis=-1).at[atom_k].set(jnp.inf)
    return jnp.min(dist)


def geometry_is_valid(positions, atom_k, graph, live):
    """Is this a geometry whose loss means anything at all?

    Only the two failures no gradient can repair: a non-finite position, and (with a live
    list) an atom with NO neighbours, which the model would score as an isolated atom — a
    spurious flat minimum arbitrarily far from the truth. Being too CLOSE is handled by the
    smooth `clash_penalty`, not by an `inf` here (an infinite plateau carries a zero
    gradient, so the optimiser learns nothing about which way to escape)."""
    valid = jnp.all(jnp.isfinite(positions[atom_k]))
    if live:
        n_atoms = positions.shape[0]
        valid = valid & jnp.any(graph.edges[keys.MASK].reshape(n_atoms, -1)[atom_k])
    return valid


def clash_penalty(d_min, cfg: SearchConfig):
    """Smooth repulsion keeping atom k out of the model's blow-up zone.

    Takes the distance to the nearest other atom (computed by the caller from the frozen
    POSITIONS — never through the neighbour finder, see the module docstring). Zero except
    within ``guard_margin`` of ``min_dist_guard``, where it grows quadratically, so its
    gradient points straight out of the clash. Added to the loss, so the selection over
    starts can never prefer a clashing geometry."""
    return cfg.guard_weight * jnp.square(
        jnp.maximum(cfg.min_dist_guard + cfg.guard_margin - d_min, 0.0)
    )


def mic_distance(a, b, cell=None):
    """Minimum-image distance between two Cartesian points (numpy, host-side)."""
    delta = np.asarray(a) - np.asarray(b)
    if cell is not None:
        frac = delta @ np.linalg.inv(cell)
        delta = (frac - np.round(frac)) @ cell
    return float(np.linalg.norm(delta))


def min_image_min_distance(pos_k, other_pos, cell=None):
    """Host-side minimum distance from pos_k to any of ``other_pos`` (min-image if periodic)."""
    delta = np.asarray(other_pos) - np.asarray(pos_k)
    if cell is not None:
        frac = delta @ np.linalg.inv(cell)
        delta = (frac - np.round(frac)) @ cell
    return float(np.linalg.norm(delta, axis=1).min())


# ── live neighbour list ─────────────────────────────────────────────────────────

def make_neighbour_rebuilder(atoms, r_max, capacity=None, probe_positions=None):
    """Return ``(rebuild, capacity)`` where ``rebuild(graph, positions) -> graph`` recomputes
    the neighbour list inside jit.

    `jax_neighbours` returns a fixed ``(n_atoms, capacity)`` array padded with ``-1``; this
    flattens it into jraph's edge layout and marks empty slots via ``edges.mask``, which
    `NequipLayer` uses to zero their contribution. Shapes never change, so nothing
    recompiles as atoms move. Membership is discrete, so no gradient flows through the
    CHOICE of neighbours — only through the edge vectors, exactly as with a frozen list, but
    now the choice is correct at every step.

    :param atoms: supplies the cell and pbc; only positions may change afterwards.
    :param r_max: the model's cutoff.
    :param capacity: slots per atom, or None to estimate and verify against ``atoms``.
    :param probe_positions: extra full ``(n_atoms, 3)`` geometries to size the capacity
        against. CRITICAL for grid search: the capacity is probed at the start geometry, but
        an atom roaming the whole cell can gain neighbours elsewhere, and an overflow SILENTLY
        drops them — corrupting the loss (a candidate's reported spectral mismatch stops being
        its true one) and manufacturing spurious minima. Pass a spread of the geometries the
        atom will actually visit so the capacity covers the worst case.
    """
    periodic = bool(np.any(atoms.pbc))
    finder = jax_neighbours.neighbour_finder(
        cutoff=float(r_max),
        cell=np.asarray(atoms.cell) if periodic else None,
        pbc=tuple(bool(x) for x in atoms.pbc) if periodic else None,
    )
    n_atoms = len(atoms)
    pos0 = jnp.asarray(atoms.positions, dtype=jnp.float32)
    probes = [pos0] + [jnp.asarray(p, dtype=jnp.float32) for p in (probe_positions or [])]

    if capacity is None:
        # The density estimate is unreliable for a small molecule (it can exceed the atom
        # count), so clamp it, then let the real geometries have the final say.
        capacity = min(int(finder.estimate_neighbours(pos0)), n_atoms)
    # Size the capacity against EVERY probe geometry, bumping to the worst case + headroom.
    # A later probe needing even more re-triggers the bump, so the loop converges.
    for pp in probes:
        probe = finder.get_neighbours(pp, max_neighbours=capacity)
        if bool(probe.did_overflow):
            capacity = int(probe.actual_max_neighbours) + 8
            probe = finder.get_neighbours(pp, max_neighbours=capacity)
            if bool(probe.did_overflow):
                raise RuntimeError(f"neighbour list still overflows at capacity {capacity}")

    def rebuild(graph, positions):
        nl = finder.get_neighbours(positions, max_neighbours=capacity)
        neighbours = nl.neighbours                       # (n_atoms, capacity), -1 = empty
        valid = neighbours != jax_neighbours.MASK_VALUE
        edges = {keys.MASK: valid.reshape(-1)}
        if keys.EDGE_CELL_SHIFTS in graph.edges:
            edges[keys.EDGE_CELL_SHIFTS] = nl.cell_indices.reshape(-1, 3).astype(jnp.float32)
        return graph._replace(
            edges=edges,
            # empty slots point at atom 0 and are masked out; `with_edge_vectors` replaces
            # masked edge vectors with 1.0, so they cannot produce NaNs in the gradient.
            senders=jnp.repeat(jnp.arange(neighbours.shape[0]), capacity),
            receivers=jnp.where(valid, neighbours, 0).reshape(-1),
            n_edge=jnp.array([neighbours.shape[0] * capacity]),
        )

    return rebuild, capacity


# ── the optimiser ────────────────────────────────────────────────────────────────

def make_multistart_adam(module, params, to_graph, atoms_base, atom_k,
                         loss_fn: Callable[[Array], Array], cfg: SearchConfig, rebuild=None):
    """Return ``run(starts) -> (best_loss, best_pos, all_losses, all_pos)`` — one Adam
    descent per start, all executed in parallel by ``jax.vmap``.

    ``loss_fn(pred_tensors) -> scalar`` is the only experiment-specific piece: it maps the
    model's ``(n_atoms, 3, 3)`` prediction to the scalar being minimised. The gradient chain
    is ``pos_k → tensors → loss_fn`` and is exact for the current topology.

    Selection is on the loss ALONE (no ground truth), and `lax.scan` keeps every step so the
    best point actually VISITED is returned, not wherever the descent happened to stop.

    Compiles once per structure: the graph shape is fixed and ``starts`` is a traced
    argument, so adding starts costs GPU width, not recompilation.
    """
    n_atoms = len(atoms_base)
    graph_base = to_graph(atoms_base)
    pos_all = jnp.asarray(atoms_base.positions, dtype=jnp.float32)
    cap = cfg.local_cap

    # Every atom but k is frozen, so the clash distance is measured against a fixed array —
    # no graph, no neighbour finder, no NaN gradient (see the module docstring).
    others = pos_all[jnp.array([i for i in range(n_atoms) if i != atom_k])]
    _cell = np.asarray(atoms_base.cell) if bool(np.any(atoms_base.pbc)) else None
    _cell_j = None if _cell is None else jnp.asarray(_cell, dtype=jnp.float32)
    _icell_j = None if _cell is None else jnp.asarray(np.linalg.inv(_cell), dtype=jnp.float32)

    def _min_dist(pos_k):
        delta = others - pos_k
        if _cell_j is not None:                      # minimum image, for periodic cells
            frac = delta @ _icell_j
            delta = (frac - jnp.round(frac)) @ _cell_j
        return jnp.linalg.norm(delta, axis=-1).min()

    def _loss(u, start):
        """The model loss, with the validity flag as aux so one forward pass serves both."""
        pos_k = bound_disp(u, start, cap, jnp)
        pos = pos_all.at[atom_k].set(pos_k)
        g = gcnn.experimental.update_graph(graph_base).set(("nodes", "positions"), pos).get()
        if rebuild is not None:
            g = rebuild(g, pos)
        out = module._model.apply(params, g)
        loss = loss_fn(out.nodes["predicted_nmr_tensors"][:n_atoms])
        return loss, geometry_is_valid(pos, atom_k, g, rebuild is not None)

    def _penalty(u, start):
        return clash_penalty(_min_dist(bound_disp(u, start, cap, jnp)), cfg)

    lrs = cfg.adam_lr1 + 0.5 * (cfg.adam_lr0 - cfg.adam_lr1) * (
        1.0 + jnp.cos(jnp.pi * jnp.arange(cfg.adam_steps) / cfg.adam_steps)
    )

    def one_run(start):
        def step(carry, lr):
            u, m, v, t = carry
            (l, ok), g = jax.value_and_grad(_loss, has_aux=True)(u, start)
            # Scrub BOTH gradients before they touch Adam's moments: one NaN is enough to
            # poison m and v permanently, and a poisoned lane emits NaN positions for the
            # rest of the run.
            g = jnp.clip(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)
            lp, gp = jax.value_and_grad(_penalty)(u, start)
            gp = jnp.clip(jnp.nan_to_num(gp, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)
            g = g + gp
            t = t + 1.0
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            u_next = u - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            total = jnp.where(ok, l + lp, jnp.inf)
            return (u_next, m, v, t), (total, bound_disp(u, start, cap, jnp))

        init = (start, jnp.zeros(3), jnp.zeros(3), jnp.array(0.0))
        _, (losses, poss) = jax.lax.scan(step, init, lrs)
        # NOT jnp.nan_to_num: its posinf default rewrites the inf sentinel to 3.4e38, a
        # finite value that then passes every isfinite() check downstream.
        losses = jnp.where(jnp.isnan(losses), jnp.inf, losses)
        i = jnp.argmin(losses)
        return losses[i], poss[i], losses, poss

    return jax.jit(jax.vmap(one_run))


def run_multistart(module, params, to_graph, atoms_base, atom_k, loss_fn, starts, cfg,
                   rebuild=None):
    """Run `make_multistart_adam` over ``starts`` (N, 3), batched to bound GPU memory.

    Returns dict(losses, positions, all_losses, all_pos), each with N leading entries; the
    invalid-geometry loss is ``inf``. A ragged final batch is padded to full width rather
    than triggering a second compilation for its shape.
    """
    starts = np.asarray(starts, dtype=np.float64)
    n = len(starts)
    run = make_multistart_adam(module, params, to_graph, atoms_base, atom_k, loss_fn, cfg,
                               rebuild=rebuild)
    batch = min(cfg.starts_per_batch, n)
    chunks = []
    for i in range(0, n, batch):
        block = starts[i:i + batch]
        pad = batch - len(block)
        if pad:
            block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
        bl, bp, al, ap = run(jnp.asarray(block, dtype=jnp.float32))
        keep = batch - pad
        chunks.append((np.asarray(bl)[:keep], np.asarray(bp)[:keep],
                       np.asarray(al)[:keep], np.asarray(ap)[:keep]))
    losses = np.concatenate([c[0] for c in chunks]).astype(np.float64)
    positions = np.concatenate([c[1] for c in chunks]).astype(np.float64)
    all_losses = np.concatenate([c[2] for c in chunks])
    all_pos = np.concatenate([c[3] for c in chunks])
    losses = np.where(np.isfinite(losses), losses, np.inf)
    return dict(losses=losses, positions=positions, all_losses=all_losses, all_pos=all_pos)


# ── start sampling ───────────────────────────────────────────────────────────────

def sample_ball_starts(centre, other_pos, n, radius, cfg: SearchConfig, rng):
    """``n`` starts uniform in VOLUME in a ball of ``radius`` around ``centre``, clash-free.

    For the local benchmark: ``centre`` is a prior on the atom's position (a perturbed site,
    or a candidate from a structural model). Nothing here needs the true position."""
    out = []
    guard = cfg.min_dist_guard + cfg.guard_margin
    for _ in range(n * 200):
        if len(out) >= n:
            break
        vec = rng.normal(size=3)
        vec /= np.linalg.norm(vec)
        cand = np.asarray(centre) + vec * radius * rng.random() ** (1.0 / 3.0)
        if np.linalg.norm(np.asarray(other_pos) - cand, axis=1).min() >= guard:
            out.append(cand)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} clash-free starts in a {radius} Å ball; "
                           f"lower min_dist_guard or the radius.")
    return np.asarray(out, dtype=np.float64)


def sample_grid_starts(atoms, atom_k, spacing, cell, cfg: SearchConfig, rng):
    """Jittered regular grid over the cell, clash-free, at target ``spacing`` (Å).

    No positional prior: a lattice in fractional coordinates (``round(|cell_i|/spacing)``
    points per axis) with a uniform ±half-sub-cell jitter per point — even coverage without
    aliasing to the crystal symmetry. Points clashing with the framework are dropped; atom
    k's own row is excluded. Returns ``(starts, n_raw)`` (accepted positions, raw grid size)."""
    lengths = np.linalg.norm(cell, axis=1)
    ns = np.maximum(1, np.round(lengths / spacing).astype(int))
    n_raw = int(np.prod(ns))
    if n_raw > cfg.max_starts:
        raise RuntimeError(f"grid would be {n_raw} points (> max_starts={cfg.max_starts}); "
                           f"raise the spacing or cfg.max_starts.")
    axes = [(np.arange(k) + 0.5) / k for k in ns]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    frac = frac + (rng.random((n_raw, 3)) - 0.5) / ns
    other_pos = np.delete(np.asarray(atoms.positions, dtype=np.float64), atom_k, axis=0)
    guard = cfg.min_dist_guard + cfg.guard_margin
    cart = frac @ cell
    keep = [p for p in cart if min_image_min_distance(p, other_pos, cell) >= guard]
    if not keep:
        raise RuntimeError("every grid point clashes with the framework; lower "
                           "min_dist_guard or the spacing.")
    return np.asarray(keep, dtype=np.float64), n_raw


# ── clustering / degeneracy ──────────────────────────────────────────────────────

def cluster_minima(losses, positions, cfg: SearchConfig, cell=None):
    """Group converged positions into distinct minima, best loss first.

    Greedy, minimum-image: walk the starts in increasing loss, opening a new cluster when a
    position is more than ``degeneracy_sep`` from every cluster centre, else attaching it to
    the nearest. Returns dicts(centre, loss, size) sorted by loss."""
    finite = np.where(np.isfinite(losses), losses, np.inf)
    clusters = []
    for i in np.argsort(finite):
        if not np.isfinite(finite[i]):
            break
        placed = False
        for c in clusters:
            if mic_distance(positions[i], c["centre"], cell) < cfg.degeneracy_sep:
                c["size"] += 1
                placed = True
                break
        if not placed:
            clusters.append(dict(centre=np.asarray(positions[i], dtype=np.float64),
                                 loss=float(finite[i]), size=1))
    return clusters


def find_degenerate_alternatives(clusters, cfg: SearchConfig, cell=None):
    """Clusters beyond the best whose loss is within ``degeneracy_tol`` (relative) of it —
    the observable degeneracy warning. Non-empty ⇒ the spectrum does not determine the site.
    Returns ``(distance_from_best, loss)`` per rival."""
    if not clusters:
        return []
    best = clusters[0]
    ceiling = best["loss"] * (1.0 + cfg.degeneracy_tol) + 1e-12
    return [(mic_distance(c["centre"], best["centre"], cell), c["loss"])
            for c in clusters[1:] if c["loss"] <= ceiling]
