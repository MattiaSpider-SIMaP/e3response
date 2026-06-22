from pathlib import Path

import numpy as np
import pytest
import reax
from tensorial.gcnn import atomic

from e3response.data.qm9_nmr import DATASET_URLS, Qm9NmrDataModule, Qm9NmrDataset

mock_dir = Path(__file__).parent / "mock_datasets" / "qm9_nmr"


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_dataset(dataset_name):
    dataset = Qm9NmrDataset(
        dataset=dataset_name,
        atom_keys=["species", "anisotropy"],
        data_dir=mock_dir,
    )
    assert len(dataset) > 0

    for i, graph in enumerate(dataset):
        assert graph is not None, f"Graph {i} is None for dataset {dataset_name}"
        assert hasattr(
            graph, "nodes"
        ), f"Graph {i} contains no attribute 'nodes' for dataset {dataset_name}"
        assert (
            "nmr_tensors" in graph.nodes
        ), f"Graph {i} lacks 'nmr_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["nmr_tensors"], np.ndarray
        ), f"'nmr_tensors' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert graph.nodes["nmr_tensors"].shape[-2:] == (
            3,
            3,
        ), f"Wrong nmr tensor shape in graph {i} for dataset {dataset_name}"
        assert (
            "nmr_tensors" in graph.nodes
        ), f"Graph {i} lacks 'nmr_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["mu"], np.ndarray
        ), f"'mu' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert (
            atomic.TOTAL_ENERGY in graph.globals
        ), f"Graph {i} missing '{atomic.TOTAL_ENERGY}' in globals for dataset {dataset_name}"
        energy = graph.globals[atomic.TOTAL_ENERGY]
        assert isinstance(
            energy, (float, np.floating, np.ndarray)
        ), f"'{atomic.TOTAL_ENERGY}' in graph {i} is not a float for dataset {dataset_name}"
        assert not np.isnan(
            energy
        ), f"'{atomic.TOTAL_ENERGY}' in graph {i} is NaN for dataset {dataset_name}"


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_datamodule(dataset_name, test_engine):
    dm = Qm9NmrDataModule(
        dataset=dataset_name, train_val_test_split=(0.6, 0.2, 0.2), batch_size=1, data_dir=mock_dir
    )

    class DummyStage(reax.Stage):
        def __init__(self):
            super().__init__(
                name="dummystage",
                module=None,
                engine=test_engine,
                rngs=test_engine.rngs,
            )

        def _step(self):
            return {}

        def log(
            self,
            name,
            value,
            batch_size=None,
            prog_bar=None,
            logger=None,
            on_step=None,
            on_epoch=None,
            reduce_fn: "reax.types.ReduceFn" = "mean",
        ):
            pass

    dm.setup(DummyStage())

    for loader_fn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = getattr(dm, loader_fn)()
        batch_tuple = next(iter(loader))

        assert isinstance(batch_tuple, tuple), f"{loader_fn} output is not a tuple"
        batch = batch_tuple[0]

        assert hasattr(batch, "nodes"), f"{loader_fn} batch has no 'nodes'"

        assert "nmr_tensors" in batch.nodes, f"{loader_fn} batch missing 'nmr_tensors'"

        nmr_tensors = batch.nodes["nmr_tensors"]

        # Shape
        assert isinstance(
            nmr_tensors, np.ndarray
        ), f"'nmr_tensors' in {loader_fn} is not a numpy array"
        assert (
            nmr_tensors.ndim == 3
        ), f"'nmr_tensors' in {loader_fn} has wrong shape {nmr_tensors.shape}"
        assert nmr_tensors.shape[-2:] == (
            3,
            3,
        ), f"Last dims of 'nmr_tensors' must be (3,3), got {nmr_tensors.shape[-2:]}"

        # Check mu
        assert "mu" in batch.nodes, f"{loader_fn} batch missing 'mu'"
        mu = batch.nodes["mu"]

        assert isinstance(mu, np.ndarray), f"'mu' in {loader_fn} is not a numpy array"
        assert mu.ndim == 1, f"'mu' in {loader_fn} has wrong shape {mu.shape}, expected 1D array"
        assert not np.any(np.isnan(mu)), f"'mu' in {loader_fn} contains NaNs"

        # Check energy
        assert hasattr(batch, "globals"), f"{loader_fn} batch has no 'globals'"
        assert (
            atomic.TOTAL_ENERGY in batch.globals
        ), f"{loader_fn} batch missing '{atomic.TOTAL_ENERGY}' in globals"
        energy = batch.globals[atomic.TOTAL_ENERGY]
        assert isinstance(energy, np.ndarray), f"'{atomic.TOTAL_ENERGY}' in {loader_fn} is not a numpy array"
        assert not np.any(np.isnan(energy)), f"'{atomic.TOTAL_ENERGY}' in {loader_fn} contains NaNs"
