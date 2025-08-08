import gzip
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Self

import h5py
import numpy as np

from .types import FeatureRecord, ObjectRecord
from .utils import get_logger

logger = get_logger(__name__)


class FileHDF5:
    def __init__(
        self,
        file_path: str,
        read_only: bool = False,
        flush_interval: int = 20,
        features_name: str = "generic",
    ) -> None:
        self.file_path = Path(file_path)
        self.read_only = read_only
        self.flush_interval = flush_interval
        self.flush_counter = 0
        self.features_name = features_name

        self.file = None
        self.ids: dict[str, int] = {}

        self.ids_dataset = None
        self.features_dataset = None

    def __enter__(self) -> Self:
        mode = "r" if self.read_only else "a"
        self.file = h5py.File(self.file_path, mode)

        # Initialize datasets if they do not exist
        self.ids_dataset = self.file.require_dataset(
            "ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.features_dataset = self.file.require_dataset(
            "features", shape=(0, 0), maxshape=(None, None), dtype=np.float32
        )
        # Load existing IDs into memory
        self.ids = {
            _id.decode(): index
            for index, _id in enumerate(self.ids_dataset.astype("S16")[:])
        }
        if not self.read_only:
            self.file.attrs["features_name"] = self.features_name

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.file is not None:
            if not self.read_only:
                self.file.flush()
            self.file.close()
            self.file = None

    def __contains__(self, _id: str) -> bool:
        return _id in self.ids

    def save(self, record: FeatureRecord, force: bool = False) -> None:
        if self.read_only:
            raise PermissionError("File is opened in read-only mode")

        if self.file is None:
            raise RuntimeError("File is not opened")

        if not force and record._id in self.ids:
            logger.info(f"Skipping save for existing ID '{record._id}'")
            return

        # Ensure datasets are initialized with the correct dimensions
        dim = len(record.feature_vector)
        if self.ids_dataset is None or self.features_dataset is None:
            self.ids_dataset = self.file.create_dataset(
                "ids",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.features_dataset = self.file.create_dataset(
                "features",
                shape=(0, dim),
                maxshape=(None, dim),
                dtype=np.float32,
            )

        # Update datasets with new record
        if record._id in self.ids:
            index = self.ids[record._id]
        else:
            index = len(self.ids)
            self.ids[record._id] = index
            self.ids_dataset.resize((index + 1,))
            self.features_dataset.resize((index + 1, dim))

        self.ids_dataset[index] = record._id.encode("utf-8")
        self.features_dataset[index, :] = np.array(
            record.feature_vector, dtype=np.float32
        )

        self.flush_counter += 1
        if self.flush_counter >= self.flush_interval:
            self.flush()

    def save_all(self, records: list[FeatureRecord], force: bool = False) -> None:
        if self.read_only:
            raise PermissionError("File is opened in read-only mode")

        if self.file is None:
            raise RuntimeError("File is not opened")

        for record in records:
            self.save(record, force=force)

        self.flush()

    def flush(self) -> None:
        if self.file is None:
            raise RuntimeError("File is not opened")

        self.file.flush()
        logger.info(f"Flushed {self.flush_counter} records to {self.file_path}")
        self.flush_counter = 0


class FileJSONL:
    def __init__(self, file_path: str, flush_interval: int = 20) -> None:
        self.file_path = Path(file_path)
        self.flush_interval = flush_interval
        self.flush_counter = 0

        self.file = None
        self.ids: set[str] = set()

        # Ensure the file exists
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
            logger.info(f"Created new file at {self.file_path}")

        # Load existing IDs if the file is not empty
        try:
            with gzip.open(self.file_path, "rt", encoding="utf-8") as file:
                self.ids = {json.loads(line)["_id"] for line in file if line.strip()}
                logger.info(f"Found {len(self.ids)} existing IDs from {self.file_path}")
        except (gzip.BadGzipFile, json.JSONDecodeError, EOFError, KeyError) as e:
            logger.error(f"Error reading existing file {self.file_path}: {e}")
            logger.info("Creating a backup file")

            backup_path = self.file_path.with_suffix(f"{self.file_path.suffix}.bak")
            shutil.copy2(self.file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")

    def __enter__(self) -> Self:
        self.file = gzip.open(self.file_path, "at", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def __contains__(self, _id: str) -> bool:
        return _id in self.ids

    def save(self, record: ObjectRecord, force: bool = False) -> None:
        if self.file is None:
            raise RuntimeError("File is not opened")

        if not force and record._id in self.ids:
            logger.info(f"Skipping save for existing ID '{record._id}'")
            return

        self.ids.add(record._id)
        filtered_record = {
            key: value for key, value in asdict(record).items() if value is not None
        }
        self.file.write(json.dumps(filtered_record, ensure_ascii=False) + "\n")

        self.flush_counter += 1
        if self.flush_counter >= self.flush_interval:
            self.flush()

    def save_all(self, records: list[ObjectRecord], force: bool = False) -> None:
        if self.file is None:
            raise RuntimeError("File is not opened")

        for record in records:
            self.save(record, force=force)

        self.flush()

    def flush(self) -> None:
        if self.file is None:
            raise RuntimeError("File is not opened")

        self.file.flush()
        logger.info(f"Flushed {self.flush_counter} records to {self.file_path}")
        self.flush_counter = 0
