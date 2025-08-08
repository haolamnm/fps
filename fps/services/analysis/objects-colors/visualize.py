import argparse
import collections
import json
import numpy as np
from skimage import io
from .extract import COLORS
from pathlib import Path
import gzip
from typing import Any
from pprint import pprint

from ...common.utils import get_logger

logger = get_logger("analysis.colors.visualize")


def get_json_by_id(input_path: Path, record_id: str) -> dict[str, Any]:
    with gzip.open(input_path, "rt", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("_id") == record_id:
                return record
    raise ValueError(f"Record with ID {record_id} not found in {input_path}")


def main(args: argparse.Namespace) -> None:
    input_json = get_json_by_id(args.input_path, args.id)
    logger.info(f"Loaded JSON file for {args.id} in {args.input_path}")
    pprint(input_json["names"])

    hwhw = np.tile((args.height, args.width), 2).reshape(1, 4)

    scores = np.array(input_json["scores"], dtype=np.float32)
    labels = np.array(input_json["names"], dtype=np.str_)
    yxyx_boxes = np.array(input_json["yxyx_boxes"], dtype=np.float32)
    yxyx_boxes = np.round(yxyx_boxes * hwhw).astype(np.int64)

    colors_per_box = collections.defaultdict(list)
    output = np.zeros((args.height, args.width, 4), dtype=np.float32)

    for box, label, score in zip(yxyx_boxes, labels, scores):
        colors_per_box[tuple(box)].append((label, score))

    for box, colors in colors_per_box.items():
        y0, x0, y1, x1 = box
        n = 1.0 / len(colors)
        for index, (label, score) in enumerate(colors):
            yi0 = int(y0 + index * n * (y1 - y0))
            yi1 = int(y0 + (index + 1) * n * (y1 - y0))
            output[yi0:yi1, x0:x1, :3] = COLORS[label]
            output[yi0:yi1, x0:x1, 3] = score

    output = (output * 255).astype(np.uint8)
    io.imsave(args.output, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="colors visualization")
    parser.add_argument(
        "input_path",
        type=Path,
        help="path to the color map file, e.g. '000-colors.jsonl.gz'",
    )
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="the ID of record to visualize, e.g. '15757-032'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./colors_visualization.png"),
        help="output path for the visualization image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="width of the visualization image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="height of the visualization image",
    )
    args = parser.parse_args()

    main(args)
