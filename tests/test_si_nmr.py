import collections
import json

import ase
from monty.json import MontyEncoder
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import reax

from e3response.data.si_nmr import SiNmrDataModule

# Mock: 3 Q classes (Q2, Q3, Q4), 4 structures each → 12 total
_MOCK_Q_CLASSES = [2, 3, 4]
_MOCK_PER_CLASS = 4
_MOCK_N_STRUCTURES = len(_MOCK_Q_CLASSES) * _MOCK_PER_CLASS


def _make_atoms(qn: int, seed: int = 0, n_atoms: int = 4, n_si: int = 1) -> ase.Atoms:
    """Minimal Atoms object with the arrays expected by SiNmrDataModule."""
    rng = np.random.default_rng(seed)
    symbols = ["Si"] * n_si + ["O"] * (n_atoms - n_si)
    atoms = ase.Atoms(symbols, positions=rng.random((n_atoms, 3)) * 2.0)
    atoms.arrays["nmr_tensors"] = rng.random((n_atoms, 3, 3))
    mask = np.zeros(n_atoms, dtype=bool)
    mask[:n_si] = True
    atoms.arrays["mask"] = mask
    atoms.arrays["Qn"] = [qn] * n_si
    return atoms


@pytest.fixture
def mock_structures() -> list[ase.Atoms]:
    """_MOCK_PER_CLASS structures per Q class, in Q-class order."""
    structures = []
    for q in _MOCK_Q_CLASSES:
        for i in range(_MOCK_PER_CLASS):
            structures.append(_make_atoms(qn=q, seed=q * 10 + i))
    return structures


class _DummyStage(reax.Stage):
    def __init__(self, engine):
        super().__init__(name="dummy", module=None, engine=engine, rngs=engine.rngs)

    def _step(self):
        return {}

    def log(self, name, value, **kwargs):
        pass


def _make_dm(split=(0.5, 0.25, 0.25)) -> SiNmrDataModule:
    return SiNmrDataModule(
        r_max=3.0,
        data_file="dummy.json",  # overridden by monkeypatch
        train_val_test_split=split,
        batch_size=1,
    )


@pytest.fixture
def mock_si_json(tmp_path):
    """A real (Monty-encoded) si_data.json with 8 minimal entries, so the actual
    `_load_structures` (and its `parse_limit` slicing) can be exercised end-to-end."""
    n_entries = 8
    entries = []
    for seed in range(n_entries):
        rng = np.random.default_rng(seed)
        structure = Structure(
            Lattice.cubic(5.0), ["Si", "O", "O", "O"], rng.random((4, 3))
        )
        entries.append(
            {
                "structure": structure,
                "ind": [0],
                "N": 4,
                "tensor": rng.random((1, 3, 3)).tolist(),
                "Qn": [3],
            }
        )
    json_path = tmp_path / "si_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, cls=MontyEncoder)
    return json_path, n_entries


@pytest.mark.parametrize(
    "limit, expected",
    [(None, 8), (3, 3), ("0:6", 6), ("2:8:2", 3)],
)
def test_si_nmr_constructor_limit(mock_si_json, limit, expected):
    """The constructor `limit` (int or slice-string) is applied via the shared
    parse_limit inside `_load_structures`."""
    json_path, _ = mock_si_json
    dm = SiNmrDataModule(r_max=3.0, data_file=json_path, batch_size=1, limit=limit)
    assert len(dm._load_structures()) == expected


def test_si_nmr_qn_grouping(mock_structures):
    """Qn grouping produces one group per Q class, each with _MOCK_PER_CLASS structures."""
    groups: dict[tuple, list] = collections.defaultdict(list)
    for atoms in mock_structures:
        label = tuple(sorted(set(atoms.arrays.get("Qn", []))))
        groups[label].append(atoms)

    assert len(groups) == len(_MOCK_Q_CLASSES), "Wrong number of Q groups"
    for q in _MOCK_Q_CLASSES:
        assert (q,) in groups, f"Missing group for Q{q}"
        assert len(groups[(q,)]) == _MOCK_PER_CLASS, f"Wrong size for Q{q} group"


def test_si_nmr_non_stratified_split(mock_structures, test_engine, monkeypatch):
    """stratify=False falls back to a plain random split over all structures (no Qn grouping):
    partitions still cover the whole dataset and are non-empty for this well-sized mock."""
    monkeypatch.setattr(SiNmrDataModule, "_load_structures", lambda self: mock_structures)

    dm = SiNmrDataModule(
        r_max=3.0,
        data_file="dummy.json",
        train_val_test_split=(0.5, 0.25, 0.25),
        batch_size=1,
        stratify=False,
    )
    dm.setup(_DummyStage(test_engine))

    n_train, n_val, n_test = len(dm.data_train), len(dm.data_val), len(dm.data_test)
    assert n_train + n_val + n_test == _MOCK_N_STRUCTURES
    assert n_train > 0 and n_val > 0 and n_test > 0


def test_si_nmr_datamodule_stratified(mock_structures, test_engine, monkeypatch):
    """Stratified split: every partition contains structures from all Q classes."""
    monkeypatch.setattr(SiNmrDataModule, "_load_structures", lambda self: mock_structures)

    dm = _make_dm(split=(0.5, 0.25, 0.25))
    dm.setup(_DummyStage(test_engine))

    n_train = len(dm.data_train)
    n_val = len(dm.data_val)
    n_test = len(dm.data_test)

    # All structures must be accounted for
    assert n_train + n_val + n_test == _MOCK_N_STRUCTURES

    # Every partition must be non-empty (guaranteed only by stratification;
    # a naive single split of 12 could leave val/test empty with a bad seed)
    assert n_train > 0, "train partition is empty"
    assert n_val > 0, "val partition is empty"
    assert n_test > 0, "test partition is empty"

    # With 4 structures per class and (0.5, 0.25, 0.25): each class → (2, 1, 1)
    # → totals (6, 3, 3) across 3 classes
    assert n_train == 6
    assert n_val == 3
    assert n_test == 3


def test_si_nmr_dataloaders(mock_structures, test_engine, monkeypatch):
    """All three dataloaders yield valid graph batches with the expected fields."""
    monkeypatch.setattr(SiNmrDataModule, "_load_structures", lambda self: mock_structures)

    dm = _make_dm()
    dm.setup(_DummyStage(test_engine))

    for loader_fn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = getattr(dm, loader_fn)()
        batch_tuple = next(iter(loader))

        assert isinstance(batch_tuple, tuple), f"{loader_fn} did not return a tuple"
        batch = batch_tuple[0]

        assert hasattr(batch, "nodes"), f"{loader_fn} batch has no 'nodes'"

        assert "nmr_tensors" in batch.nodes, f"{loader_fn} batch missing 'nmr_tensors'"
        nmr = batch.nodes["nmr_tensors"]
        assert isinstance(nmr, np.ndarray), f"'nmr_tensors' in {loader_fn} is not ndarray"
        assert nmr.ndim == 3, f"'nmr_tensors' ndim wrong in {loader_fn}"
        assert nmr.shape[-2:] == (3, 3), f"'nmr_tensors' shape wrong in {loader_fn}"

        assert "nmr_active" in batch.nodes, f"{loader_fn} batch missing 'nmr_active'"
        mask = batch.nodes["nmr_active"]
        assert isinstance(mask, np.ndarray), f"'nmr_active' in {loader_fn} is not ndarray"
        assert mask.dtype == bool, f"'nmr_active' in {loader_fn} is not bool"
