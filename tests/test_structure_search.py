"""Tests for the shared structure-resolution search engine (`e3response.structure_search`).

Everything here is model-free: the pieces that need a trained checkpoint (the Adam descent
itself) are exercised by the demo scripts' parity checks, not in unit tests. What is tested
here is the loss-agnostic machinery — the geometry guards, the start sampling, the
clustering, and (the tricky one) the live neighbour rebuilder against `graph_from_ase`.
"""
import numpy as np
import jax.numpy as jnp
import pytest
from ase import Atoms
from tensorial import gcnn

from e3response import keys, structure_search as ss


# ── config / small helpers ───────────────────────────────────────────────────────

def test_bound_disp_caps_and_is_identity():
    x0 = jnp.zeros(3)
    # far-away u is pulled inside the cap
    assert float(jnp.linalg.norm(ss.bound_disp(jnp.array([10.0, 0, 0]), x0, 1.5))) < 1.5
    # tiny step is essentially unchanged
    small = jnp.array([0.01, 0.0, 0.0])
    assert float(jnp.linalg.norm(ss.bound_disp(small, x0, 1.5) - small)) < 1e-4
    # None disables the cap
    assert float(jnp.linalg.norm(ss.bound_disp(jnp.array([10.0, 0, 0]), x0, None))) == 10.0


def test_mic_distance_periodic():
    cell = np.eye(3) * 5.0
    # 4.5 Å apart in the cell is 0.5 Å under the minimum image
    assert ss.mic_distance([0.0, 0, 0], [4.5, 0, 0], cell) == pytest.approx(0.5)
    # without a cell it is the plain distance
    assert ss.mic_distance([0.0, 0, 0], [4.5, 0, 0], None) == pytest.approx(4.5)


def test_clash_penalty_zero_far_positive_near():
    cfg = ss.SearchConfig(min_dist_guard=0.5, guard_margin=0.15, guard_weight=200.0)
    assert float(ss.clash_penalty(jnp.asarray(2.0), cfg)) == 0.0          # far: no penalty
    assert float(ss.clash_penalty(jnp.asarray(0.5), cfg)) == pytest.approx(200.0 * 0.15**2)
    # monotonically increasing as the atoms get closer
    near = float(ss.clash_penalty(jnp.asarray(0.3), cfg))
    assert near > float(ss.clash_penalty(jnp.asarray(0.5), cfg))


# ── clustering / degeneracy ──────────────────────────────────────────────────────

def test_cluster_minima_merges_and_orders():
    cfg = ss.SearchConfig(degeneracy_sep=0.5)
    losses = np.array([1.0, 1.02, 5.0, np.inf])
    pos = np.array([[0, 0, 0], [0.1, 0, 0], [3, 0, 0], [9, 9, 9]], dtype=float)
    clusters = ss.cluster_minima(losses, pos, cfg)
    assert len(clusters) == 2                       # the two close low-loss points merge
    assert clusters[0]["loss"] == 1.0 and clusters[0]["size"] == 2
    assert clusters[1]["loss"] == 5.0               # inf is dropped, order by loss


def test_find_degenerate_alternatives():
    cfg = ss.SearchConfig(degeneracy_tol=0.25)
    # a rival within 25% of the best loss, far away → flagged
    near = [dict(centre=np.zeros(3), loss=1.0, size=3),
            dict(centre=np.array([2.0, 0, 0]), loss=1.2, size=2)]
    assert len(ss.find_degenerate_alternatives(near, cfg)) == 1
    # a rival well above the tolerance → not flagged
    far = [dict(centre=np.zeros(3), loss=1.0, size=3),
           dict(centre=np.array([2.0, 0, 0]), loss=2.0, size=2)]
    assert ss.find_degenerate_alternatives(far, cfg) == []


# ── start sampling ───────────────────────────────────────────────────────────────

def test_sample_ball_starts_count_and_clash_free():
    cfg = ss.SearchConfig(min_dist_guard=0.5, guard_margin=0.15)
    rng = np.random.default_rng(0)
    other = np.array([[3.0, 3, 3]])                 # one far atom, nothing to clash with
    starts = ss.sample_ball_starts([0.0, 0, 0], other, 16, 2.0, cfg, rng)
    assert starts.shape == (16, 3)
    assert np.linalg.norm(starts - np.array([0.0, 0, 0]), axis=1).max() <= 2.0 + 1e-9
    guard = cfg.min_dist_guard + cfg.guard_margin
    assert np.linalg.norm(other - starts[:, None], axis=2).min() >= guard


def test_sample_grid_starts_jittered_and_clash_free():
    cfg = ss.SearchConfig(min_dist_guard=0.5, guard_margin=0.15, max_starts=10000)
    cell = np.eye(3) * 6.0
    # a single framework atom at the centre; grid points near it must be rejected
    atoms = Atoms("He2", positions=[[3.0, 3, 3], [0.1, 0.1, 0.1]], cell=cell, pbc=True)
    rng = np.random.default_rng(1)
    starts, n_raw = ss.sample_grid_starts(atoms, atom_k=1, spacing=2.0, cell=cell,
                                          cfg=cfg, rng=rng)
    assert n_raw == 3 * 3 * 3                        # round(6/2)=3 per axis
    assert 0 < len(starts) <= n_raw                 # some points rejected near the atom
    guard = cfg.min_dist_guard + cfg.guard_margin
    for p in starts:                                 # every accepted start is clash-free
        assert ss.min_image_min_distance(p, [[3.0, 3, 3]], cell) >= guard


def test_sample_grid_starts_respects_max_starts():
    cfg = ss.SearchConfig(max_starts=8)
    cell = np.eye(3) * 20.0
    atoms = Atoms("He", positions=[[0.0, 0, 0]], cell=cell, pbc=True)
    with pytest.raises(RuntimeError, match="max_starts"):
        ss.sample_grid_starts(atoms, 0, spacing=1.0, cell=cell, cfg=cfg,
                              rng=np.random.default_rng(0))


# ── live neighbour rebuilder vs graph_from_ase ────────────────────────────────────

def _edge_multiset(senders, receivers, mask=None):
    s = np.asarray(senders); r = np.asarray(receivers)
    if mask is not None:
        keep = np.asarray(mask).astype(bool)
        s, r = s[keep], r[keep]
    return sorted(zip(s.tolist(), r.tolist()))


def test_rebuilder_matches_graph_from_ase_open():
    # A non-periodic cluster: the rebuilt edge set must equal graph_from_ase's exactly.
    rng = np.random.default_rng(2)
    pos = rng.normal(scale=1.5, size=(6, 3))
    atoms = Atoms("C6", positions=pos)              # no cell/pbc → open boundary
    r_max = 2.5
    to_graph = lambda a: gcnn.atomic.graph_from_ase(a, r_max=r_max,
                                                    atom_include_keys=("numbers",))
    graph = to_graph(atoms)
    rebuild, cap = ss.make_neighbour_rebuilder(atoms, r_max)
    g2 = rebuild(graph, jnp.asarray(atoms.positions, dtype=jnp.float32))

    ref = _edge_multiset(graph.senders, graph.receivers)
    got = _edge_multiset(g2.senders, g2.receivers, g2.edges[keys.MASK])
    assert got == ref


def test_rebuilder_matches_graph_from_ase_periodic():
    # A periodic cell: edge COUNT must match (pairs recur with different cell shifts, so
    # compare the number of edges rather than the bare sender/receiver multiset).
    cell = np.eye(3) * 4.0
    atoms = Atoms("C4", scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
                  cell=cell, pbc=True)
    r_max = 2.2
    to_graph = lambda a: gcnn.atomic.graph_from_ase(
        a, r_max=r_max, atom_include_keys=("numbers",))
    graph = to_graph(atoms)
    rebuild, cap = ss.make_neighbour_rebuilder(atoms, r_max)
    g2 = rebuild(graph, jnp.asarray(atoms.positions, dtype=jnp.float32))
    n_ref = int(graph.senders.shape[0])
    n_got = int(np.asarray(g2.edges[keys.MASK]).astype(bool).sum())
    assert n_got == n_ref
