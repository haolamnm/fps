import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord
from ...common.utils import get_logger

logger = get_logger("analysis.yolo.extract")


def load_image_pil(image_path: Path) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"), dtype=np.float32)
            width, height = image.size

        return image_np, width, height
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}") from e


def convert_xyxy_to_yxyx_boxes(
    xyxy_boxes: np.ndarray, image_width: int, image_height: int
) -> np.ndarray:
    xyxy_boxes_np = np.array(xyxy_boxes, dtype=np.float32)

    # Make sure the input is a 2D array
    if len(xyxy_boxes_np.shape) == 1:
        xyxy_boxes_np = xyxy_boxes_np.reshape(1, -1)

    # Extract coordinates and normalize them
    x1, y1, x2, y2 = (
        xyxy_boxes_np[:, 0],
        xyxy_boxes_np[:, 1],
        xyxy_boxes_np[:, 2],
        xyxy_boxes_np[:, 3],
    )
    x1_norm = x1 / image_width
    y1_norm = y1 / image_height
    x2_norm = x2 / image_width
    y2_norm = y2 / image_height

    yxyx_boxes = np.column_stack([y1_norm, x1_norm, y2_norm, x2_norm])
    return yxyx_boxes


def get_record(results: Any, image_width: int, image_height: int) -> ObjectRecord:
    try:
        xyxy_boxes = results.boxes.xyxy.cpu().numpy()
        yxyx_boxes: list[tuple[float, float, float, float]] = (
            convert_xyxy_to_yxyx_boxes(xyxy_boxes, image_width, image_height).tolist()
        )
        scores: list[float] = results.boxes.conf.cpu().numpy().tolist()
        labels: list[int] = results.boxes.cls.cpu().numpy().tolist()
        names: list[str] = [results.names[int(label)] for label in labels]

        return ObjectRecord(
            _id="",
            detector="yolov11",
            names=names,
            labels=labels,
            yxyx_boxes=yxyx_boxes,
            scores=scores,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to parse detection results: {e}") from e


class YoloExtractor(BaseObjectExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-version",
            type=str,
            default="yolo11x.pt",
            choices=[
                "yolo11n.pt",
                "yolo11s.pt",
                "yolo11m.pt",
                "yolo11l.pt",
                "yolo11x.pt",
            ],
            help="yolo model to use",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.model_version: str = args.model_version
        self.device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"

        model_path = Path(__file__).parent / "checkpoint" / self.model_version
        self.model = YOLO(model_path).to(self.device).eval()
        logger.info(f"Loaded model from {model_path}")

        self.model.compile()
        logger.info("Compiled model for performance optimization")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[ObjectRecord]:
        for frame_path in frame_paths:
            try:
                image_np, width, height = load_image_pil(frame_path)
                with torch.no_grad():
                    result = self.model(image_np, verbose=False)[0]
                record = get_record(result, width, height)
                record._id = frame_path.stem

                logger.info(
                    f"Processed {record._id} with {len(record.labels or [])} labels"
                )
                yield record
            except Exception as e:
                logger.error(f"Failed to process {frame_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yolo extractor")
    YoloExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = YoloExtractor(args)
    extractor.run()
