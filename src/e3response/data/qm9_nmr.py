import collections
from collections.abc import Callable, Sequence
import functools
from functools import lru_cache
import logging
import os
import pathlib
import re
import tempfile
from typing import Any, Callable, Final, Optional, Sequence, Union

import urllib.error
import urllib.request
import zipfile

import ase
from flax import nnx
import jraph
import numpy as np
from pymatgen.io import gaussian  # type: ignore
import pymatgen.io.ase  # type: ignore
import reax
from tensorial import gcnn
from tensorial.gcnn import atomic
import tqdm
from typing_extensions import override

from e3response import keys

__all__ = ("Qm9NmrDataset", "Qm9NmrDataModule")

_LOGGER = logging.getLogger(__name__)

# Archive paths whose integrity has already been verified in this process, so the
# expensive ``testzip`` scan runs at most once per archive (see `_ensure_archive`).
_VALIDATED_ARCHIVES: set[str] = set()


# QM9 NMR datasets
DATASET_URLS = {
    "gasphase": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=dwVDQQTtRGC5V5OH1Ddbpg",
    "CCl4": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=ly5xV6JXRpuwa9ByWP-a4w",
    "THF": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=PKMdIIOsQR644mo2PIvPIg",
    "acetone": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=RhoELQmVS2K0AxPHW0JFbw",
    "methanol": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=cMfYU0u1RcuA6P9uqwQXng",
    "DMSO": "https://nomad-lab.eu/prod/rae/api/raw/query?dataset_id=417HCiXDRhC22th2aE4Xzw",
}

# Nuclear magnetic moments dict
mu_dict = {
    "H": 2.792847351,  # 1H
    "C": 0.702369,  # 13C
    "N": -0.2830569,  # 15N
    "O": -1.893543,  # 17O
    "F": 2.628321,  # 19F
}


def _parse_limit(limit: int | str | None) -> slice:
    """Convert a limit spec to a slice over a sorted file list.

    - None       → slice(None)       (all files)
    - int N      → slice(None, N)    (first N files)
    - "a:b"      → slice(a, b)       (files a through b-1)
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


class Qm9NmrDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """
    QM9-NMR dataset in different solvents containing graphs
    with full NMR tensors and related quantities (optional).
    """

    def __init__(
        self,
        r_max: float = 5,
        data_dir: str | pathlib.Path = "data/qm9_nmr/",
        dataset: str | Sequence[str] = "gasphase",
        atom_keys: str | Sequence[str] | None = None,
        limit: int | str | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """
        Initialize the QM9-NMR dataset.

        :param r_max: Maximum cutoff radius for graph construction.
        :param data_dir: Directory where dataset archives are stored.
        :param dataset: List of dataset names containing gaussian raw data.
        :param tensors: Name(s) of tensor(s) to extract, either a string
                        (for one tensor) or a list/tuple of strings.
        :param limit: Controls which structures to load from the archive
            (files are sorted by name before slicing, so indices are stable):
            - None  → all structures
            - int N → first N structures (equivalent to ``"0:N"``)
            - str "start:stop" or "start:stop:step" → Python-slice semantics,
              e.g. ``"10000:10020"`` loads only those 20 structures and the
              resulting dataset has indices 0–19.
        :param indices: If given, restrict the dataset to exactly these positions
            within the `limit`-selected file list (e.g. the scattered indices of one
            train/val/test split — see `Qm9NmrDataModule.load_split`). Only the
            selected ``.log`` files are read/parsed. ``None`` (default) loads everything in
            `limit`.
        """
        super().__init__()

        if isinstance(dataset, str):
            self.dataset = [dataset]
        else:
            self.dataset = list(dataset)

        for ds in self.dataset:
            if ds not in DATASET_URLS:
                raise ValueError(
                    f"Dataset '{ds}' not recognised. Available: {list(DATASET_URLS.keys())}"
                )

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        # Params
        self._rmax = r_max
        self._data_dir: Final[str] = data_dir
        self._limit = limit
        default_keys = ["nmr_tensors", "mu"]
        possible_keys = [
            "ind",
            "N",
            "species",
            "isotropic",
            "anisotropy",
            "eigenvalues",
        ]

        if isinstance(atom_keys, str):
            atom_keys = [atom_keys]

        invalid_keys = [key for key in (atom_keys or []) if key not in possible_keys]
        if invalid_keys:
            raise ValueError(
                f"Invalid atom_keys: {invalid_keys}. " f"Allowed keys are: {possible_keys}"
            )

        self._atom_keys = list(set(default_keys).union(atom_keys or []))

        self._to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
            gcnn.atomic.graph_from_ase,
            r_max=self._rmax,
            atom_include_keys=("numbers", *self._atom_keys),
            global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, atomic.TOTAL_ENERGY],
        )

        # Data: collect (archive_path, log_file) pairs across all requested archives,
        # honouring `limit` per archive (as before), then optionally sub-select
        # `indices` BEFORE reading/parsing anything.
        archive_log_pairs: list[tuple[str, str]] = []
        for ds in self.dataset:
            archive_path = self._ensure_archive(data_dir, ds)
            log_files = self._list_log_files(archive_path, limit=self._limit)
            archive_log_pairs.extend((archive_path, log_file) for log_file in log_files)

        if indices is not None:
            archive_log_pairs = [archive_log_pairs[i] for i in indices]

        self._data = self._extract_log_files(archive_log_pairs)
        self._data_tuple = tuple(self._data)

        @lru_cache(maxsize=100000)
        def _get_graph_worker(index):
            return self._to_graph(self._data[index])

        self._get_graph_worker = _get_graph_worker

    def __getitem__(self, index):
        return self._get_graph_worker(index)

    def __len__(self):
        return len(self._data)

    def cache_info(self):
        return self._get_graph_worker.cache_info()

    def clear_cache(self):
        self._get_graph_worker.cache_clear()

    @staticmethod
    def _download_file(name: str, url: str, path: str) -> None:
        _LOGGER.info("\nDownloading %s from %s ...", name, url)

        try:
            with tqdm.tqdm(unit="B", unit_scale=True, desc=os.path.basename(path)) as progress_bar:

                def reporthook(_block_num, block_size, total_size):
                    if progress_bar.total is None and total_size > 0:
                        progress_bar.total = total_size
                    progress_bar.update(block_size)

                urllib.request.urlretrieve(url, filename=path, reporthook=reporthook)  # nosec B310

            _LOGGER.info("\nDownload completed: %s", path)

        except urllib.error.URLError as e:
            _LOGGER.error("Network error during download of %s: %s", name, e)

        except OSError as e:
            _LOGGER.error("Filesystem error while writing %s: %s", path, e)

    @classmethod
    def _ensure_archive(cls, data_dir: str | pathlib.Path, ds: str) -> str:
        """Return a valid local path to the archive for dataset `ds`, downloading (or
        re-downloading, if corrupted) it as needed.

        The (potentially expensive) ``testzip`` integrity check is run at most once per
        archive per process — subsequent calls for an already-validated, still-present
        file skip it — so repeated instantiations (e.g. ``count`` followed by the actual
        load, or loading several splits in a row) don't re-scan the whole archive."""
        archive_name = f"QM9nmr_{ds}_logs.zip"
        archive_path = os.path.join(data_dir, archive_name)
        url = DATASET_URLS[ds]

        if archive_path in _VALIDATED_ARCHIVES and os.path.isfile(archive_path):
            return archive_path

        if os.path.isfile(archive_path):
            try:
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.testzip()
            except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as e:
                _LOGGER.warning(
                    "%s is corrupted or unreadable: %s, removing corrupted archive ...",
                    archive_name,
                    e,
                )
                os.remove(archive_path)
                cls._download_file(archive_name, url, archive_path)
            else:
                _LOGGER.info("%s already present and valid at %s.", archive_name, archive_path)
        else:
            _LOGGER.info("%s not found.", archive_name)
            cls._download_file(archive_name, url, archive_path)

        _VALIDATED_ARCHIVES.add(archive_path)
        return archive_path

    @staticmethod
    def _list_log_files(zip_path: str, limit: int | str | None = None) -> list[str]:
        """List the sorted, `limit`-sliced ``.log`` filenames in `zip_path`, without
        reading or parsing any of them."""
        limit_slice = _parse_limit(limit)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Sort so that integer-range limits have stable, reproducible semantics.
            log_files = sorted(f for f in zip_ref.namelist() if f.endswith(".log"))
        return log_files[limit_slice]

    @staticmethod
    def _extract_log_files(archive_log_pairs: Sequence[tuple[str, str]]) -> list:
        """Read and parse exactly the given `(archive_path, log_file)` pairs."""
        structures = []
        open_zips: dict[str, zipfile.ZipFile] = {}
        try:
            for archive_path, log_file in tqdm.tqdm(archive_log_pairs, desc="EXTRACT ZIP"):
                zip_ref = open_zips.get(archive_path)
                if zip_ref is None:
                    zip_ref = zipfile.ZipFile(archive_path, "r")
                    open_zips[archive_path] = zip_ref

                data = zip_ref.read(log_file)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".log", encoding="utf-8"
                ) as tmp_log:
                    tmp_log.write(data.decode("utf-8"))
                    # flush before reading back by path, or the log may still be empty
                    tmp_log.flush()
                    structures.append(get_structure_and_data_from_log(pathlib.Path(tmp_log.name)))
        finally:
            for zip_ref in open_zips.values():
                zip_ref.close()

        return structures

    @classmethod
    def count(
        cls,
        data_dir: str | pathlib.Path,
        dataset: str | Sequence[str] = "gasphase",
        limit: int | str | None = None,
    ) -> int:
        """Cheaply count how many structures `(dataset, limit)` selects, without parsing
        any ``.log`` files. Useful for computing a train/val/test split ahead of loading,
        so only the wanted split's files need to be extracted (see
        `Qm9NmrDataModule.load_split`)."""
        names = [dataset] if isinstance(dataset, str) else list(dataset)
        return sum(
            len(cls._list_log_files(cls._ensure_archive(data_dir, ds), limit=limit))
            for ds in names
        )


# pylint: disable=R1710
def _create_molecule_data(log_file):
    try:
        gaussian_output = gaussian.GaussianOutput(log_file)

        # check for structure
        if len(gaussian_output.structures) == 0:
            raise ValueError(f"File {log_file} does not contain final structure.")

        structure = gaussian_output.final_structure

        # extraction of data from .log file
        with open(log_file, encoding="utf-8") as file:
            log_data = file.read()

        shielding_pattern = (
            r"(\d+)\s+"  # atom index
            r"([A-Za-z])\s+"  # element symbol
            r"Isotropic\s+=\s+([-\d\.]+)\s+"  # isotropic shielding
            r"Anisotropy\s+=\s+([-\d\.]+)\s+"  # anisotropy
            r"XX=\s+([-\d\.]+)\s+"  # tensor component XX
            r"YX=\s+([-\d\.]+)\s+"  # YX
            r"ZX=\s+([-\d\.]+)\s+"  # ZX
            r"XY=\s+([-\d\.]+)\s+"  # XY
            r"YY=\s+([-\d\.]+)\s+"  # YY
            r"ZY=\s+([-\d\.]+)\s+"  # ZY
            r"XZ=\s+([-\d\.]+)\s+"  # XZ
            r"YZ=\s+([-\d\.]+)\s+"  # YZ
            r"ZZ=\s+([-\d\.]+)\s+"  # ZZ
            r"Eigenvalues:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"  # eigenvalues
        )

        energy_pattern = r"SCF Done:\s+E\([^)]+\)\s+=\s+([-\d\.]+)\s+A\.U\."
        energy_match = re.search(energy_pattern, log_data)
        if energy_match is None:
            raise ValueError(f"File {log_file} does not contain SCF energy.")
        energy = float(energy_match.group(1))

        matches = re.findall(shielding_pattern, log_data)

        atom_list = []
        for match in matches:
            (
                atom_number,
                atom_type,
                isotropic,
                anisotropy,
                *tensor_vals,
                eigenvalue1,
                eigenvalue2,
                eigenvalue3,
            ) = match

            tensor_matrix = np.array([float(x) for x in tensor_vals]).reshape(3, 3)

            atom_list.append(
                {
                    "index": int(atom_number),
                    "species": atom_type,
                    "tensor": tensor_matrix,
                    "isotropic": float(isotropic),
                    "anisotropy": float(anisotropy),
                    "eigenvalues": [float(eigenvalue1), float(eigenvalue2), float(eigenvalue3)],
                }
            )

        # final dictionary
        molecule_data = {
            "structure": structure,
            "energy": energy,
            **{
                key: [atom[key] for atom in atom_list]
                for key in ["tensor", "isotropic", "anisotropy", "eigenvalues", "species"]
            },
            "ind": list(range(len(structure))),
            "N": len(structure),
        }

        return molecule_data

    except ValueError as e:
        _LOGGER.error("Error in file %s: %s", log_file, e)
        raise

    except OSError as e:
        _LOGGER.error("File system error while processing %s: %s", log_file, e)
        raise


def get_structure_and_data_from_log(log_path: pathlib.Path) -> ase.Atoms | None:
    # _LOGGER.info("Parsing Gaussian .log file: %s", log_path)

    try:
        molecule_data = _create_molecule_data(log_path)
        if molecule_data is None:
            _LOGGER.warning("No valid structure in %s", log_path.name)
            return None

        atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(molecule_data["structure"])
        assert isinstance(atoms, pymatgen.io.ase.Atoms)

        ind = molecule_data["ind"]
        n_atoms = molecule_data["N"]
        assert isinstance(n_atoms, int), "The number of atoms is not an integer."

        tensors = np.zeros((n_atoms, 3, 3))
        tensors[ind] = molecule_data["tensor"]

        atoms.arrays["nmr_tensors"] = tensors
        atoms.arrays["ind"] = np.array(ind)
        atoms.arrays["N"] = np.array(n_atoms)
        atoms.arrays["species"] = np.array(molecule_data["species"])
        atoms.arrays["isotropic"] = np.array(molecule_data["isotropic"])
        atoms.arrays["anisotropy"] = np.array(molecule_data["anisotropy"])
        atoms.arrays["eigenvalues"] = np.array(molecule_data["eigenvalues"])

        species = molecule_data["species"]
        mu_values = np.array([mu_dict[s] for s in species])
        atoms.arrays["mu"] = mu_values
        atoms.arrays[keys.EXTERNAL_MAGNETIC_FIELD] = np.zeros(3)
        atoms.arrays[atomic.TOTAL_ENERGY] = np.array(molecule_data["energy"])

        # print(atoms.arrays["mu"])

        return atoms

    except (ValueError, OSError) as e:
        _LOGGER.error("Parsing error for %s: %s", log_path, e)
    return None


class Qm9NmrDataModule(reax.DataModule):
    """
    QM9-NMR data module containing graphs with full NMR tensors
    and related quantities subdivided in train/val/test and batches.
    """

    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float = 5,
        data_dir: str | pathlib.Path = "data/qm9_nmr/",
        dataset: str | Sequence[str] = "gasphase",
        atom_keys: Sequence[str] | None = None,
        limit: int | str | None = None,
        train_val_test_split: Sequence[int | float] = (0.85, 0.05, 0.1),
        batch_size: int = 64,
    ) -> None:
        """Initialize a QM9-NMR data module.

        :param r_max: Maximum cutoff radius for graph construction.
        :param data_dir: Directory where dataset archives are stored.
        :param dataset: List of dataset names containing gaussian raw data.
        :param tensors: Name(s) of tensor(s) to extract, either a string
                        (for one tensor) or a list/tuple of strings.
        :param limit: Maximum number of structures to load as graphs.
        :param train_val_test_split: The train, validation and test split.
        :param batch_size: The batch size. Defaults to 64.
        """
        super().__init__()

        # Params
        self._data_dir: Final[pathlib.Path] = pathlib.Path(data_dir)
        self._dataset: str | Sequence[str] = dataset
        self.dataset: Qm9NmrDataset | None = None
        self._rmax = r_max
        self._atom_keys = atom_keys
        self._limit = limit
        self._train_val_test_split: Final[Sequence[int | float]] = train_val_test_split
        self._batch_size: Final[int] = batch_size

        # State
        self.batch_size_per_device = batch_size
        self.data_train: reax.data.Dataset | None = None
        self.data_val: reax.data.Dataset | None = None
        self.data_test: reax.data.Dataset | None = None

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        """Load data. Set variables: self.data_train, self.data_val, self.data_test.

        This method is called by REAX before trainer.fit(), trainer.validate(),
        trainer.test(), and trainer.predict(), so be careful not to execute things like random
        split twice! Also, it is called after self.prepare_data() and there is a barrier in
        between which ensures that all the processes proceed to self.setup() once the data is
        prepared and available for use.

        :param stage: The stage to setup. Either "fit", "validate", "test", or "predict".
        Defaults to `None.
        """

        if self.dataset is None:
            self.dataset = Qm9NmrDataset(
                r_max=self._rmax,
                data_dir=self._data_dir,
                dataset=self._dataset,
                atom_keys=self._atom_keys,
                limit=self._limit,
            )

        # load and split dataset only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            # Split up the graphs into sets
            train, val, test = reax.data.random_split(
                stage.rngs, dataset=self.dataset, lengths=self._train_val_test_split
            )

            calc_padding = functools.partial(
                gcnn.data.GraphBatcher.calculate_padding,
                batch_size=self._batch_size,
                with_shuffle=True,
            )

            paddings = list(map(calc_padding, (train, val, test)))
            # Calculate the max padding we will need for any of the batches
            self._max_padding = gcnn.data.max_padding(*paddings)

            self.data_train = train
            self.data_val = val
            self.data_test = test

    def load_split(
        self,
        split: str,
        limit: int | str | None = None,
        rngs: "nnx.Rngs | None" = None,
    ) -> "Qm9NmrDataset":
        """Load only the requested split ("train"/"val"/"test"), extracting just its
        ``.log`` files.

        Unlike `setup()` — which must extract the full `limit`-restricted archive
        because training needs all three splits at once — this only parses the files
        belonging to the requested split. Useful for post-hoc analysis (e.g. recovering
        exactly which structures were held out at test time for an already-trained run)
        without paying for the rest of the dataset.

        :param split: which partition to load: "train", "val" or "test".
        :param limit: like `Qm9NmrDataset`'s own `limit`, but applied to the split's
            indices instead of the whole dataset — e.g. `limit=20` loads only the
            first 20 structures of the split, `"10:30"` loads structures 10-29 of it.
            `None` (default) loads the whole split.
        :param rngs: must match whatever was used at training time to reproduce the
            SAME split; defaults to `nnx.Rngs(0)`, REAX's own default when no
            `Trainer`/`Engine` override is given (true for every config in this repo).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        if rngs is None:
            rngs = nnx.Rngs(0)

        n = Qm9NmrDataset.count(self._data_dir, self._dataset, self._limit)
        splits = reax.data.random_split(rngs, dataset=range(n), lengths=self._train_val_test_split)
        indices = dict(zip(("train", "val", "test"), splits))[split].indices
        indices = indices[_parse_limit(limit)]

        return Qm9NmrDataset(
            r_max=self._rmax,
            data_dir=self._data_dir,
            dataset=self._dataset,
            atom_keys=self._atom_keys,
            limit=self._limit,
            indices=indices,
        )

    @override
    def train_dataloader(self) -> reax.DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )
