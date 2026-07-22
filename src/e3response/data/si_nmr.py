import collections
import functools
import json
import logging
import pathlib
from typing import Any, Final, Optional, Sequence, Union

from ase import Atoms
from flax import nnx
import jraph
from monty.json import MontyDecoder
import numpy as np
import pymatgen.io.ase
import reax
from tensorial import gcnn
from typing_extensions import override

from e3response import keys

_LOGGER = logging.getLogger(__name__)

__all__ = ("SiNmrDataModule",)


def _parse_limit(limit: int | str | None) -> slice:
    """Convert a limit spec to a slice over a split's structure list.

    - None       → slice(None)       (all structures)
    - int N      → slice(None, N)    (first N structures)
    - "a:b"      → slice(a, b)       (structures a through b-1)
    - "a:b:s"    → slice(a, b, s)    (with step)
    """
    if limit is None:
        return slice(None)
    if isinstance(limit, int):
        return slice(None, limit)
    parts = limit.split(":")
    indices = [int(p) if p else None for p in parts]
    if len(indices) == 2:
        return slice(indices[0], indices[1])
    if len(indices) == 3:
        return slice(indices[0], indices[1], indices[2])
    raise ValueError(
        f"Cannot parse limit {limit!r}: expected int, 'start:stop', or 'start:stop:step'"
    )


class SiNmrDataModule(reax.DataModule):
    """Silicon dataset containing ASE atoms objects with nmr tensors per Silicon atom, nmr active atoms mask annd Q_n classification."""

    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float,
        data_file: Union[str, pathlib.Path] = "data/si_nmr/si_data.json",
        train_val_test_split: Sequence[Union[int, float]] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Params
        self._rmax: Final[float] = r_max
        self._data_file: Final[str] = str(data_file)
        self._train_val_test_split: Final[Sequence[Union[int, float]]] = train_val_test_split
        self._batch_size: Final[int] = batch_size
        self._limit = limit

        # State
        self.batch_size_per_device = batch_size
        self.data_train: Optional[reax.data.Dataset] = None
        self.data_val: Optional[reax.data.Dataset] = None
        self.data_test: Optional[reax.data.Dataset] = None

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        if self.data_train is not None:
            return

        structures = self._load_structures()
        train, val, test = self._grouped_split(structures, stage.rngs)

        train_graphs = list(map(self._to_graph, train))
        val_graphs = list(map(self._to_graph, val))
        test_graphs = list(map(self._to_graph, test))

        calc_padding = functools.partial(
            gcnn.data.GraphBatcher.calculate_padding, batch_size=self._batch_size, with_shuffle=True
        )

        self._max_padding = gcnn.data.max_padding(
            *map(calc_padding, (train_graphs, val_graphs, test_graphs))
        )

        self.data_train = train_graphs
        self.data_val = val_graphs
        self.data_test = test_graphs

    def _grouped_split(
        self, structures: list[Atoms], rngs: "nnx.Rngs"
    ) -> tuple[list[Atoms], list[Atoms], list[Atoms]]:
        """Stratified train/val/test split: groups structures by their Qn signature
        and splits each group independently so every partition contains the same
        ratio of Q classes."""
        groups: dict[tuple, list] = collections.defaultdict(list)
        for atoms in structures:
            label = tuple(sorted(set(atoms.arrays.get("Qn", []))))
            groups[label].append(atoms)

        train, val, test = [], [], []
        for group in groups.values():
            g_train, g_val, g_test = reax.data.random_split(
                rngs, dataset=group, lengths=self._train_val_test_split
            )
            train.extend(list(g_train))
            val.extend(list(g_val))
            test.extend(list(g_test))
        return train, val, test

    def _to_graph(self, atoms: Atoms) -> jraph.GraphsTuple:
        return gcnn.atomic.graph_from_ase(
            atoms,
            r_max=self._rmax,
            atom_include_keys=("numbers", "nmr_tensors", "mask"),
            global_include_keys=[],
            key_mapping={"mask": "nmr_active"},
        )

    def load_split(
        self,
        split: str,
        limit: Optional[Union[int, str]] = None,
        rngs: "nnx.Rngs | None" = None,
    ) -> list[jraph.GraphsTuple]:
        """Load only the requested split ("train"/"val"/"test") as graphs, without
        running `setup()` or building the other splits.

        Useful for post-hoc analysis (e.g. recovering exactly which structures were
        held out at test time for an already-trained run).

        :param split: which partition to load: "train", "val" or "test".
        :param limit: further restricts the returned split - int N takes the first N
            structures of the split, "start:stop"/"start:stop:step" applies Python-slice
            semantics. `None` (default) returns the whole split.
        :param rngs: must match whatever was used at training time to reproduce the
            SAME split; defaults to `nnx.Rngs(0)`, REAX's own default when no
            `Trainer`/`Engine` override is given (true for every config in this repo).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        if rngs is None:
            rngs = nnx.Rngs(0)

        structures = self._load_structures()
        train, val, test = self._grouped_split(structures, rngs)
        split_structures = dict(zip(("train", "val", "test"), (train, val, test)))[split]
        split_structures = split_structures[_parse_limit(limit)]

        return list(map(self._to_graph, split_structures))

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())

        with open(path, encoding="utf-8") as f:
            entries = json.load(f, cls=MontyDecoder)

        structures = []
        for entry in entries:
            atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(entry["structure"])

            ind = entry["ind"]
            n_atoms = entry["N"]

            tensors = np.zeros((n_atoms, 3, 3))
            tensors[ind] = entry["tensor"]

            mask = np.zeros(n_atoms, dtype=bool)
            mask[ind] = True

            atoms.arrays["nmr_tensors"] = tensors
            atoms.arrays["mask"] = mask
            atoms.arrays["Qn"] = entry["Qn"]

            structures.append(atoms)

        if self._limit is not None:
            structures = structures[: self._limit]

        _LOGGER.info("Number of loaded structures: %d", len(structures))
        return structures

    @override
    def train_dataloader(self) -> reax.DataLoader:
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader:
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader:
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )
