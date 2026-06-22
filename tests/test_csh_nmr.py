import pickle
from pathlib import Path

import ase
import numpy as np
import pytest
import reax

from e3response.data.csh_nmr import CshNmrDataModule

# Mock dataset constants: 5 bulk + 5 surface, cutoff overridden via monkeypatch
_MOCK_BULK_CUTOFF = 5
_MOCK_N_STRUCTURES = 10


def _make_atoms(seed: int = 0, n_atoms: int = 3) -> ase.Atoms:
    """Minimal Atoms object with the arrays expected by CshNmrDataModule."""
    rng = np.random.default_rng(seed)
    atoms = ase.Atoms(
        "H" * n_atoms,
        positions=rng.random((n_atoms, 3)) * 2.0,  # kept within r_max=3.0
    )
    atoms.arrays["nmr_tensors"] = rng.random((n_atoms, 3, 3))
    atoms.arrays["mask"] = np.ones(n_atoms, dtype=bool)
    return atoms


@pytest.fixture
def mock_csh_pkl(tmp_path: Path) -> Path:
    """Pickle file with 10 fake Atoms objects (indices 0-4 = bulk, 5-9 = surface)."""
    structures = [_make_atoms(seed=i) for i in range(_MOCK_N_STRUCTURES)]
    pkl_path = tmp_path / "csh_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(structures, f)
    return pkl_path


class _DummyStage(reax.Stage):
    def __init__(self, engine):
        super().__init__(name="dummy", module=None, engine=engine, rngs=engine.rngs)

    def _step(self):
        return {}

    def log(self, name, value, **kwargs):
        pass


def test_csh_nmr_datamodule_full_dataset_stratified(mock_csh_pkl, test_engine, monkeypatch):
    """Full dataset: both bulk and surface structures appear in every partition."""
    monkeypatch.setattr(CshNmrDataModule, "_BULK_CUTOFF", _MOCK_BULK_CUTOFF)

    dm = CshNmrDataModule(
        r_max=3.0,
        data_file=mock_csh_pkl,
        train_val_test_split=(0.6, 0.2, 0.2),
        batch_size=1,
    )
    dm.setup(_DummyStage(test_engine))

    n_train = len(dm.data_train)
    n_val = len(dm.data_val)
    n_test = len(dm.data_test)

    # All structures must be accounted for
    assert n_train + n_val + n_test == _MOCK_N_STRUCTURES

    # Every partition must be non-empty (guaranteed by stratification)
    assert n_train > 0, "train partition is empty"
    assert n_val > 0, "val partition is empty"
    assert n_test > 0, "test partition is empty"

    # With stratified 60/20/20 on 5 bulk + 5 surface:
    # each group gives (3, 1, 1) → totals (6, 2, 2)
    assert n_train == 6
    assert n_val == 2
    assert n_test == 2


def test_csh_nmr_datamodule_only_bulk(mock_csh_pkl, test_engine, monkeypatch):
    """limit='only bulk' loads only the first _BULK_CUTOFF structures."""
    monkeypatch.setattr(CshNmrDataModule, "_BULK_CUTOFF", _MOCK_BULK_CUTOFF)

    dm = CshNmrDataModule(
        r_max=3.0,
        data_file=mock_csh_pkl,
        train_val_test_split=(0.6, 0.2, 0.2),
        batch_size=1,
        limit="only bulk",
    )
    dm.setup(_DummyStage(test_engine))

    total = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert total == _MOCK_BULK_CUTOFF, (
        f"Expected {_MOCK_BULK_CUTOFF} structures with 'only bulk', got {total}"
    )


def test_csh_nmr_dataloaders(mock_csh_pkl, test_engine, monkeypatch):
    """All three dataloaders yield valid graph batches with the expected fields."""
    monkeypatch.setattr(CshNmrDataModule, "_BULK_CUTOFF", _MOCK_BULK_CUTOFF)

    dm = CshNmrDataModule(
        r_max=3.0,
        data_file=mock_csh_pkl,
        train_val_test_split=(0.6, 0.2, 0.2),
        batch_size=1,
    )
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
        assert nmr.ndim == 3, f"'nmr_tensors' in {loader_fn} has wrong ndim {nmr.ndim}"
        assert nmr.shape[-2:] == (3, 3), (
            f"'nmr_tensors' last dims must be (3,3), got {nmr.shape[-2:]}"
        )
