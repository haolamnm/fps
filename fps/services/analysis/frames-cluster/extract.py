import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from ...common.files import FileJSONL
from ...common.types import ObjectRecord
from ...common.utils import get_logger

logger = get_logger(__name__)


@np.vectorize
def _ascii_encode(index: int, ascii_range: tuple[int, int] = (33, 126)) -> str:
    """Converts an integer to a two-character string using printable ASCII characters (codes 33-126)."""

    ascii_min, ascii_max = ascii_range
    base = ascii_max - ascii_min + 1

    first_char = chr(ascii_min + (index // base))
    second_char = chr(ascii_min + (index % base))

    return first_char + second_char


def cluster(frame_features: np.ndarray) -> list[str]:
    num_samples = frame_features.shape[0]

    if num_samples == 1:
        return ["!!"]

    if num_samples >= 94**2:
        logger.warning("Exceeding maximum number of clusters (8836)")

    labels = []
    distance_matrix = squareform(pdist(frame_features, metric="euclidean"))
    thrs = np.arange(0.35, 1.50, 0.05)

    for thr in thrs:
        assignments = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thr,
            linkage="single",
        ).fit_predict(distance_matrix)

        labels.append(assignments)

        if len(np.unique(assignments)) == 1:
            logger.warning(f"All samples assigned to the same cluster at threshold {thr:.2f}")
            break

    labels = np.column_stack(labels)
    codes = _ascii_encode(labels)
    codes: list[str] = ["".join(code) for code in codes]
    return codes


def main(args: argparse.Namespace) -> None:
    logger.info("Starting clustering extraction")

    with h5py.File(args.features_file, "r") as file:
        frame_ids = np.array(file["ids"].asstr()[:], dtype=np.str_)  # type: ignore
        frame_features = np.array(file["features"][:], dtype=np.float32)  # type: ignore

    cluster_codes = cluster(frame_features)

    records = [
        ObjectRecord(
            _id=frame_id,
            detector="cluster",
            cluster_code=cluster_code,
        )
        for frame_id, cluster_code in zip(frame_ids, cluster_codes)
    ]

    if args.force and args.output_file.exists():
        logger.info(f"Overwriting existing output file {args.output_file}")
        args.output_file.unlink()

    with FileJSONL(args.output_file) as file:
        file.save_all(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster extractor")
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="force extraction even if output file exists",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        dest="features_file",
        required=True,
        help="path to the HDF5 file containing frame features",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_file",
        required=True,
        help="path to the output JSONL file for storing cluster codes",
    )
    args = parser.parse_args()

    main(args)
