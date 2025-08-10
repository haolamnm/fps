import argparse
import functools
import itertools
import random
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import numpy.typing as npt
import surrogate

from ...common.files import FileJSONL
from ...common.types import ObjectRecord
from ...common.utils import get_logger, load_config

logger = get_logger(__name__)


def get_train_features(
    features_file_template: Path, video_ids: list[str]
) -> tuple[int, str, Iterator[npt.NDArray[np.float32]]]:
    # Peek features name
    features_file = str(features_file_template).format(video_id=video_ids[0])
    with h5py.File(features_file, "r") as file:
        features_name = str(file.attrs["features_name"])
        features_dim = int(file["features"].shape[1])  # type: ignore

    # Generate training features
    def get_all_features() -> Iterator[npt.NDArray[np.float32]]:
        for video_id in video_ids:
            features_file = str(features_file_template).format(video_id=video_id)
            with h5py.File(features_file, "r") as file:
                yield from file["features"]  # type: ignore

    train_features = get_all_features()
    return features_dim, features_name, train_features


def load_encoder(
    encoder_file_path: Path,
    encoder_config: dict[str, Any],
    features_dim: int,
    train_features: Iterator[npt.NDArray[np.float32]],
    num_trains: int,
    force: bool = False,
) -> Any:
    # Create STR encoder if not exists or force is True
    if not encoder_file_path.exists() or force:
        # Get encoder parameters
        default_index_type = "topk-sq"
        default_index_params = {
            "keep": 0.25,
            "dim_multiplier": 3.0,
        }
        index_type: str = encoder_config.get("index_type", default_index_type)
        index_params: dict[str, float] = encoder_config.get("index_params", default_index_params)

        # Init the encoder
        index_string = ", ".join(f"{key}={value}" for key, value in index_params.items())
        index_string = f"{index_type}({index_string})"
        logger.info(f"Building STR encoder: {index_string} for {features_dim} dimensions")

        encoder = surrogate.index_factory(features_dim, index_type, index_params)

        if not encoder.is_trained:
            logger.info("Collecting training features")
            train_features = itertools.islice(train_features, 0, num_trains)
            random.shuffle(list(train_features))

            logger.info("Training STR encoder")
            encoder.train(list(train_features))
            logger.info("Completed training STR encoder")

        # Save the encoder
        surrogate.save_index(encoder, encoder_file_path)
        logger.info(f"Saved STR encoder to {encoder_file_path}")

    # Load the encoder
    else:
        encoder = surrogate.load_index(encoder_file_path)
        logger.info(f"Loaded STR encoder from {encoder_file_path}")

    return encoder


def process_video_id(
    features_path: Path,
    output_path: Path,
    encoder: Any,
    force: bool = False,
    flush_interval: int = 20,
) -> None:
    with h5py.File(features_path, "r") as file:
        if output_path.exists():
            if force:
                output_path.unlink()
            else:
                logger.info(f"Skipping STR features encoding for {features_path} as output already exists")
                return

        ids = np.array(file["ids"].asstr()[:], dtype=np.str_)  # type: ignore
        features = np.array(file["features"][:], dtype=np.float32)  # type: ignore

    # Open output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with FileJSONL(str(output_path), flush_interval) as file:
        mask = [_id not in file for _id in ids]
        ids = ids[mask]
        features = features[mask]

        num_remaining = len(ids)
        batch_size: int = args.batch_size

        # Create batches of features
        batches = (features[i : i + batch_size] for i in range(0, num_remaining, batch_size))

        # Encode and generate surrogate documents in batches
        batch_encode = functools.partial(encoder.encode, inverted=False)
        encoded_batches = map(batch_encode, batches)
        str_batches = map(surrogate.generate_documents, encoded_batches)
        str_encodings = itertools.chain.from_iterable(str_batches)

        # Generate records
        records = [
            ObjectRecord(
                _id=str(_id),
                detector="str",
                feature_str=str_encoding,
            )
            for _id, str_encoding in zip(ids, str_encodings)
        ]
        file.save_all(records)


def main(args: argparse.Namespace) -> None:
    # Peek features name and dimension
    features_dim, features_name, train_features = get_train_features(args.features_file_template, args.video_ids)

    # Load config
    encoder_config = load_config(args.config_path)["index"]["features"][features_name]
    encoder = load_encoder(
        args.encoder_file_path,
        encoder_config,
        features_dim,
        train_features,
        args.train_size,
        force=args.force,
    )

    for video_id in args.video_ids:
        features_file = Path(str(args.features_file_template).format(video_id=video_id))
        output_file = Path(str(args.output_file_template).format(video_id=video_id))
        process_video_id(
            features_file,
            output_file,
            encoder,
            args.force,
            args.flush_interval,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="str-features encoder")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("~/fps/config.yaml").expanduser(),
        help="path to the configuration file",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=20,
        help="number of items to flush to disk at once",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="force re-encoding of STR features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="number of features to process in a batch",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500_000,
        help="number of training features to use for STR encoder",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        required=True,
        help="list of video IDs to process",
    )

    parser.add_argument(
        "features_file_template",
        type=Path,
        help="template for the features file path, e.g. '{video_id}.h5'",
    )
    parser.add_argument(
        "encoder_file_path",
        type=Path,
        help="path to save the STR encoder, e.g 'encoder.pkl'",
    )
    parser.add_argument(
        "output_file_template",
        type=Path,
        help="template for the output file path, e.g. '{video_id}.jsonl.gz'",
    )
    args = parser.parse_args()

    main(args)
