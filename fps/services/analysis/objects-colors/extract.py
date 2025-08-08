import argparse
import collections
import itertools
import multiprocessing
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage import io, measure, transform

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord
from ...common.utils import get_logger

logger = get_logger("analysis.colors.extract")


def load_image(image_path: Path) -> np.ndarray:
    try:
        image_np = io.imread(image_path)

        # Convert to RGB if necessary
        if image_np.ndim == 2:  # Grayscale image_np
            image_np = np.stack([image_np] * 3, axis=-1)
            logger.info(f"Converted grayscale image {image_path} to RGB")
        elif image_np.shape[2] == 4:  # RGBA image_np
            image_np = image_np[:, :, :3]
            logger.info(f"Converted RGBA image {image_path} to RGB")

        return image_np

    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")


def extract_colors(
    image_np: np.ndarray,
    color_map: np.ndarray,
    num_rows: int = 7,
    num_cols: int = 7,
    dominant_threshold: float = 0.30,
    associated_threshold: float = 0.15,
    quotient_threshold: float = 0.30,
    dominant_only: bool = False,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    # Map whole image to color index, color quantization
    image_index = (image_np // 8).astype(np.uint16)
    image_index *= np.array([1, 32, 1024], dtype=np.uint16).reshape((1, 1, 3))
    image_index = image_index.sum(axis=2)
    image_index = color_map[image_index]

    im_height = image_np.shape[0]
    im_width = image_np.shape[1]

    tile_height = im_height // num_rows
    tile_width = im_width // num_cols

    tiles_colors: dict[tuple[int, int], list[tuple[int, float]]] = {}

    for r in range(num_rows):
        for c in range(num_cols):
            tile = image_index[
                r * tile_height : (r + 1) * tile_height,
                c * tile_width : (c + 1) * tile_width,
            ]

            # Find areas per color index
            # Shift color indexes, as 0 is a reserved label (ignore) for regionprops
            tile = tile + 1
            props = measure.regionprops_table(tile, properties=("label", "area"))

            # Shift back to original color index
            color_areas: npt.NDArray[np.floating] = props["area"] / tile.size
            color_labels: npt.NDArray[np.integer] = props["label"] - 1

            # Identify dominant color
            dominant_index = color_areas.argmax()
            dominant_color: int = color_labels[dominant_index]
            dominant_area: float = color_areas[dominant_index]

            tile_colors: list[tuple[int, float]] = []

            if dominant_area > dominant_threshold:
                tile_colors.append((dominant_color, dominant_area))

                # If dominant_only is False, find associated colors
                if not dominant_only:
                    is_associated = (
                        (color_areas >= associated_threshold)
                        & ((color_areas / dominant_area) >= quotient_threshold)
                    ).astype(bool)
                    is_associated[dominant_index] = False

                    associated_colors = color_labels[is_associated]
                    associated_areas = color_areas[is_associated]

                    tile_colors.extend(zip(associated_colors, associated_areas))

            tile_colors.sort(key=lambda x: x[1], reverse=True)
            tiles_colors[(r, c)] = tile_colors

    return tiles_colors


def merge_colors(
    tables: list[dict[tuple[int, int], list[tuple[int, float]]]],
    keep_duplicates: bool = True,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    def merge_cells(cells: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        num_tables = len(cells)
        chained_cells = itertools.chain.from_iterable(cells)

        if not keep_duplicates:
            out: dict[int, float] = collections.defaultdict(float)
            for color, score in chained_cells:
                out[color] += score / num_tables
            chained_cells = out.items()
        return sorted(chained_cells, key=lambda x: x[1], reverse=True)

    keys = tables[0].keys()
    merged_table = {key: merge_cells([t[key] for t in tables]) for key in keys}

    return merged_table


def convert_table_to_record(
    color_table: dict[tuple[int, int], list[tuple[int, float]]],
    label_map: list[str],
    num_rows: int,
    num_cols: int,
) -> ObjectRecord:
    scores: list[float] = []
    yxyx_boxes: list[tuple[float, float, float, float]] = []
    labels: list[str] = []

    for (r, c), cell_colors in color_table.items():
        if not cell_colors:
            continue

        # yxyx format
        yxyx_bbox = (r / num_rows, c / num_cols, (r + 1) / num_rows, (c + 1) / num_cols)

        cell_labels, cell_scores = zip(*cell_colors)
        cell_labels = [label_map[c] for c in cell_labels]

        scores.extend(cell_scores)
        labels.extend(cell_labels)
        yxyx_boxes.extend([yxyx_bbox] * len(cell_colors))

    return ObjectRecord(
        _id="",
        scores=scores,
        yxyx_boxes=yxyx_boxes,
        names=labels,
        detector="colors",
    )


def compute_monochromaticity(image_np: Any, eps: float = 1e-7) -> float:
    """Based on https://stackoverflow.com/a/59218331/3175629"""

    image_np = transform.resize(image_np, (128, 128))  # Downsample
    pixels = image_np.reshape(-1, 3)  # List of RGB pixels
    pixels -= pixels.mean(axis=0)  # Center on mean pixel

    dd = np.linalg.svd(pixels, compute_uv=False)  # Get variance in the 3 PCA directions
    var1: float = dd[0] / (dd.sum() + eps)  # Expaned variance in first direction

    # var1 is 0 if all pixels are the same color, set to 1 in this case
    return var1 or 1.0


COLORS = {
    "black": [0.00, 0.00, 0.00],
    "blue": [0.00, 0.00, 1.00],
    "brown": [0.50, 0.40, 0.25],
    "gray": [0.50, 0.50, 0.50],
    "green": [0.00, 1.00, 0.00],
    "orange": [1.00, 0.80, 0.00],
    "pink": [1.00, 0.50, 1.00],
    "purple": [1.00, 0.00, 1.00],
    "red": [1.00, 0.00, 0.00],
    "white": [1.00, 1.00, 1.00],
    "yellow": [1.00, 1.00, 0.00],
}
LABEL_MAP = list(COLORS.keys())


class ColorsExtractor(BaseObjectExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num-rows", type=int, default=7)
        parser.add_argument("--num-cols", type=int, default=7)
        parser.add_argument("--dominant-threshold", type=float, default=0.30)
        parser.add_argument("--associated-threshold", type=float, default=0.15)
        parser.add_argument("--quotient-threshold", type=float, default=0.30)
        parser.add_argument("--dominant-only", action="store_true", default=False)
        parser.add_argument("--keep-duplicates", action="store_true", default=False)
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.num_rows: int = args.num_rows
        self.num_cols: int = args.num_cols
        self.dominant_threshold: float = args.dominant_threshold
        self.associated_threshold: float = args.associated_threshold
        self.quotient_threshold: float = args.quotient_threshold
        self.dominant_only: bool = args.dominant_only
        self.keep_duplicates: bool = args.keep_duplicates

        num_colors = len(COLORS)
        column_names = ["R", "G", "B"] + list(range(num_colors))

        def read_color_table(path: Path) -> np.ndarray:
            color_table = pd.read_csv(
                path, names=column_names, index_col=["R", "G", "B"], sep=r"\s+"
            )
            pixel_color_mapping = pd.Series(color_table.idxmax(axis=1)).to_numpy()
            logger.info(f"Loaded color table from {path}")
            return pixel_color_mapping

        tables_dir = Path(__file__).parent / "tables"
        self.josa_map = read_color_table(tables_dir / "LUT_JOSA.txt")
        self.w2c_map = read_color_table(tables_dir / "w2c.txt")

    def extract_path(self, frame_path: Path) -> ObjectRecord:
        image_np = load_image(frame_path)

        josa_colors = extract_colors(
            image_np,
            color_map=self.josa_map,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            dominant_threshold=self.dominant_threshold,
            associated_threshold=self.associated_threshold,
            quotient_threshold=self.quotient_threshold,
            dominant_only=self.dominant_only,
        )
        w2c_colors = extract_colors(
            image_np,
            color_map=self.w2c_map,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            dominant_threshold=self.dominant_threshold,
            associated_threshold=self.associated_threshold,
            quotient_threshold=self.quotient_threshold,
            dominant_only=self.dominant_only,
        )
        color_table = merge_colors(
            [josa_colors, w2c_colors], keep_duplicates=self.keep_duplicates
        )
        record = convert_table_to_record(
            color_table, LABEL_MAP, self.num_rows, self.num_cols
        )
        record.monochrome = compute_monochromaticity(image_np)
        record._id = frame_path.stem
        return record

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[ObjectRecord]:
        chunk_size = multiprocessing.cpu_count()

        with multiprocessing.Pool() as pool:
            for record in pool.imap_unordered(
                self.extract_path, frame_paths, chunksize=chunk_size
            ):
                logger.info(f"Extracted colors for {record._id}")
                yield record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="colors extractor")
    ColorsExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = ColorsExtractor(args)
    extractor.run()
