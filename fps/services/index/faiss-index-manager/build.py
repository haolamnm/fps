import argparse
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import faiss
import h5py
import more_itertools
import numpy as np

from ...common.utils import get_logger, load_config

logger = get_logger(__name__)


def peek_file_attributes(file_path: Path) -> tuple[int, str]:
    with h5py.File(file_path, "r") as file:
        if "features" not in file:
            raise ValueError(f"File {file_path} does not contain 'features' dataset.")
        features_dataset = file["features"]
        if not isinstance(features_dataset, h5py.Dataset):
            raise ValueError(f"'features' in {file_path} is not a dataset.")
        features_dim = int(features_dataset.shape[1])
        features_name = str(file.attrs["features_name"])
        if not features_name:
            raise ValueError(f"'features_name' attribute in {file_path} is empty.")

    return features_dim, features_name


def load_ids_and_features(file_paths: list[Path]) -> Iterator[tuple[str, np.ndarray]]:
    for file_path in file_paths:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping.")
            continue
        with h5py.File(file_path, "r") as file:
            if "features" not in file:
                logger.warning(f"File {file_path} does not contain 'features' dataset, skipping.")
                continue
            if "ids" not in file:
                logger.warning(f"File {file_path} does not contain 'ids' dataset, skipping.")
                continue
            features_dataset = file["features"]
            if not isinstance(features_dataset, h5py.Dataset):
                logger.warning(f"'features' in {file_path} is not a dataset, skipping.")
                continue
            ids_dataset = file["ids"]
            if not isinstance(ids_dataset, h5py.Dataset):
                logger.warning(f"'ids' in {file_path} is not a dataset, skipping.")
                continue

            ids = np.array(ids_dataset.asstr()[:], dtype=np.str_)
            features = np.array(features_dataset[:], dtype=np.float32)

            yield from zip(ids, features, strict=False)


def create(args: argparse.Namespace) -> None:
    # Skip if existing index and ID map files are present
    if not args.force and args.index_path.exists() and args.idmap_path.exists():
        logger.info(f"Index file {args.index_path} and ID map file {args.idmap_path} already exist")
        logger.info("Use --force to overwrite the existing index and ID map files")
        return

    # Load ids and features from the specified directory
    features_files: list[Path] = args.features_dir.glob("*.h5")
    features_files = sorted(features_files, key=lambda x: x.name)
    ids_and_features = load_ids_and_features(features_files)

    # Peek at the first file to get the feature dimension and name
    if not features_files:
        logger.error("No feature files found in the specified directory.")
        return
    features_dim, features_name = peek_file_attributes(features_files[0])

    # Load configuration
    config = load_config(args.config_path)["index"]["features"][features_name]
    index_type: str = config["index_type"]

    # Create index
    logger.info(f"Creating {index_type} index with dimension {features_dim}")
    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(features_dim, index_type, metric)

    # Train the index if necessary
    if not index.is_trained:
        logger.info(f"Training index with {args.train_size} samples")
        training_samples = more_itertools.take(args.train_size, ids_and_features)
        training_features = np.array([feature for _, feature in training_samples], dtype=np.float32)
        index.train(training_features)
        logger.info("Completed training the index")

    # Add features to the index in batches
    logger.info(f"Adding features to the index in batches of size {args.batch_size}")
    batches = more_itertools.batched(ids_and_features, args.batch_size)

    with open(args.idmap_path, "w") as idmap_file:
        for batch in batches:
            ids_batch, features_batch = zip(*batch, strict=True)
            idmap_file.write("\n".join(ids_batch) + "\n")  # type: ignore

            features_batch = np.stack(features_batch).astype(np.float32)
            index.add(features_batch)

    logger.info(f"Created index with {index.ntotal} features")

    faiss.write_index(index, str(args.index_path))
    logger.info(f"Saved index to {args.index_path}")


def add(args: argparse.Namespace) -> None:
    if args.index_path.exists() and args.idmap_path.exists():
        # Load existing index and ID map
        index = faiss.read_index(str(args.index_path))
        with open(args.idmap_path) as lines:
            idmap = list(map(str.rstrip, lines.readlines()))

    # Else, there is no index, create an empty one
    else:
        features_dim, features_name = peek_file_attributes(args.features_files[0])

        # Load configuration
        config = load_config(args.config_path)["index"]["features"][features_name]
        index_type: str = config["index_type"]

        # Create an empty index
        logger.info(f"Creating empty {index_type} index with dimension {features_dim}")
        metric = faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(features_dim, index_type, metric)

        idmap: list[str] = []

    assert index.is_trained, "Index must be trained before adding features"

    def _add_features_file(features_file: Path, index: Any, idmap: list[str]) -> tuple[Any, list[str]]:
        """Add features from a single file to the index."""

        # Load ids and features from the file
        with h5py.File(features_file, "r") as file:
            ids = np.array(file["ids"].asstr()[:], dtype=np.str_)  # type: ignore
            positions = [index for index, _id in enumerate(idmap) if _id in ids]
            if not args.force and positions:
                logger.info(f"Skipping adding features from {features_file} as they already exist in the index")
                return index, idmap

            features = np.array((file["features"][:]), dtype=np.float32)  # type: ignore

        # If force, remove existing features from the index
        if args.force and positions:
            idmap = [_id for index, _id in enumerate(idmap) if index not in positions]
            index.remove_ids(faiss.IDSelectorBatch(np.array(positions, dtype=np.int64)))  # type: ignore

        # Add new features to the index in batches
        ids_and_features = zip(ids, features, strict=True)
        batches = more_itertools.batched(ids_and_features, args.batch_size)

        for batch in batches:
            ids_batch, features_batch = zip(*batch, strict=True)
            idmap.extend(ids_batch)
            features_batch = np.stack(features_batch).astype(np.float32)
            index.add(features_batch)

        return index, idmap

    for features_file in args.features_files:
        if not features_file.exists():
            logger.warning(f"Feature file {features_file} does not exist, skipping.")
            continue
        index, idmap = _add_features_file(features_file, index, idmap)

    logger.info(f"Updated index with {index.ntotal} features")

    with open(args.idmap_path, "w") as idmap_file:
        idmap_file.write("\n".join(idmap) + "\n")

    faiss.write_index(index, str(args.index_path))
    logger.info(f"Saved index to {args.index_path}")


def remove(args: argparse.Namespace) -> None:
    if not args.index_path.exists() or not args.idmap_path.exists():
        logger.error("Index or ID map file does not exist. Cannot remove features.")
        return

    # Load index and ID map
    index = faiss.read_index(str(args.index_path))
    with open(args.idmap_path) as lines:
        idmap = list(map(str.rstrip, lines.readlines()))

    positions: set[int] = set()
    for video_id in args.video_ids:
        regexp = re.compile(re.escape(video_id) + r"-\d+")
        positions.update([index for index, _id in enumerate(idmap) if regexp.match(_id)])

    # Skip index saving if no changes are made
    if not positions:
        logger.info("Made no changes to the index, no features removed.")
        return

    idmap = [_id for index, _id in enumerate(idmap) if index not in positions]
    positions = np.fromiter(positions, dtype=np.int64, count=len(positions))  # type: ignore
    index.remove_ids(faiss.IDSelectorBatch(positions))  # type: ignore
    logger.info(f"Removed {len(positions)} features from the index")

    # Save updated index and ID map
    with open(args.idmap_path, "w") as idmap_file:
        idmap_file.write("\n".join(idmap) + "\n")

    faiss.write_index(index, str(args.index_path))
    logger.info(f"Saved updated index to {args.index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faiss builder")
    parser.add_argument(
        "--config-path",
        default=Path("~/fps/config.yaml").expanduser(),
        type=Path,
        help="path to yaml configuration file",
    )
    parser.add_argument(
        "index_path",
        type=Path,
        help="path to the output index file",
    )
    parser.add_argument(
        "idmap_path",
        type=Path,
        help="path to the output ID map file",
    )

    subparsers = parser.add_subparsers(help="command")

    create_parser = subparsers.add_parser("create", help="create index")
    create_parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="force overwrite existing index and ID map files",
    )
    create_parser.add_argument(
        "--batch-size",
        default=50_000,
        type=int,
        help="batch size for adding features to the index",
    )
    create_parser.add_argument(
        "--train-size",
        default=50_000,
        type=int,
        help="number of elements to use for training the index",
    )
    create_parser.add_argument(
        "features_dir",
        type=Path,
        help="directory containing feature files",
    )
    create_parser.set_defaults(func=create)

    add_parser = subparsers.add_parser("add", help="add features to index")
    add_parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="force add features even if index already exists",
    )
    add_parser.add_argument(
        "--batch-size",
        default=50_000,
        type=int,
        help="batch size for adding features to the index",
    )
    add_parser.add_argument(
        "features_files",
        nargs="+",
        type=Path,
        help="list of feature files to add to the index",
    )
    add_parser.set_defaults(func=add)

    remove_parser = subparsers.add_parser("remove", help="remove a video from the index")
    remove_parser.add_argument(
        "video_ids",
        nargs="+",
        type=str,
        help="ID(s) of the video(s) to remove from the index",
    )
    remove_parser.set_defaults(func=remove)

    args = parser.parse_args()
    args.func(args)
