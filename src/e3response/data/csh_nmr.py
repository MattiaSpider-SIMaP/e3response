import collections
import json
import logging
import pathlib
from typing import Any, Callable, Final, Optional, Sequence, Union

from ase import Atoms
import jraph
import numpy as np
import reax
from tensorial import gcnn
from typing_extensions import override

from e3response import keys

_LOGGER = logging.getLogger(__name__)

__all__ = ("CshNmrDataModule",)


class CshNmrDataModule(reax.DataModule):
    """Calcium Silicate Hydrate dataset from a pre-processed JSON file containing ASE atoms objects with NMR tensor data."""

    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float,
        data_file: Union[str, pathlib.Path] = "data/csh_nmr/Na_csh.json",
        train_val_test_split: Sequence[Union[int, float]] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        limit: Optional[Union[int, str]] = None,
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

        # Group structures by struct_type ("bulk"/"surf") and split each group
        # independently so every partition contains the same ratio of each type.
        groups: dict[str, list[Atoms]] = collections.defaultdict(list)
        for atoms in structures:
            groups[atoms.info.get("struct_type")].append(atoms)

        train, val, test = [], [], []
        for group in groups.values():
            g_train, g_val, g_test = reax.data.random_split(
                stage.rngs, dataset=group, lengths=self._train_val_test_split
            )
            train.extend(list(g_train))
            val.extend(list(g_val))
            test.extend(list(g_test))

        to_graph: Callable[[Atoms], jraph.GraphsTuple] = lambda atoms: gcnn.atomic.graph_from_ase(
            atoms,
            r_max=self._rmax,
            atom_include_keys=("numbers", "nmr_tensors", "mask"),
            global_include_keys=[],
        )

        train_graphs = list(map(to_graph, train))
        val_graphs = list(map(to_graph, val))
        test_graphs = list(map(to_graph, test))

        calc_padding = lambda graphs: gcnn.data.GraphBatcher.calculate_padding(
            graphs, batch_size=self._batch_size, with_shuffle=True
        )

        self._max_padding = gcnn.data.max_padding(
            *map(calc_padding, (train_graphs, val_graphs, test_graphs))
        )

        self.data_train = train_graphs
        self.data_val = val_graphs
        self.data_test = test_graphs

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())

        with open(path, encoding="utf-8") as file:
            entries = json.load(file)

        structures = [self._entry_to_atoms(entry) for entry in entries]

        if isinstance(self._limit, str):
            limit_lower = self._limit.lower()
            if limit_lower == "only bulk":
                structures = [s for s in structures if s.info.get("struct_type") == "bulk"]
            elif limit_lower == "only surface":
                structures = [s for s in structures if s.info.get("struct_type") == "surf"]
            else:
                raise ValueError(f"Unknown limit option: {self._limit}")
        elif self._limit is not None:
            structures = structures[: self._limit]

        _LOGGER.info("Number of loaded structures: %d", len(structures))

        return structures

    @staticmethod
    def _entry_to_atoms(entry: dict[str, Any]) -> Atoms:
        atoms = Atoms(
            numbers=entry["numbers"],
            positions=entry["positions"],
            cell=entry["cell"],
            pbc=entry["pbc"],
        )
        atoms.arrays["nmr_tensors"] = np.asarray(entry["nmr_tensors"])
        atoms.arrays["mask"] = np.asarray(entry["mask"], dtype=bool)
        atoms.info["struct_type"] = entry.get("struct_type")
        atoms.info["ca_si_ratio"] = entry.get("ca_si_ratio")
        atoms.info["energy_Ry"] = entry.get("energy_Ry")
        return atoms

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
