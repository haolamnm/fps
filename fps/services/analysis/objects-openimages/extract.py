import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord
from ...common.utils import get_logger

logger = get_logger(__name__)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


def load_image_pil(image_path: Path) -> np.ndarray:
    """Load an image using PIL."""
    try:
        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"), dtype=np.float32)

        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
        image_np = image_np.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return image_np
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}") from e


def get_record(detection_raw: dict[str, tf.Tensor]) -> ObjectRecord:
    detection_data: dict[str, list] = {
        key: value.numpy().tolist()  # type: ignore
        for key, value in detection_raw.items()
    }

    for field in ("detection_class_names", "detection_class_entities"):
        if field in detection_data:
            detection_data[field] = [
                label.decode("utf-8") if isinstance(label, bytes) else label
                for label in detection_data[field]
            ]

    record = ObjectRecord(
        _id="",
        detector="frcnn-oiv4",
        labels=detection_data.get("detection_class_labels", []),
        entities=detection_data.get("detection_class_names", []),  #
        names=detection_data.get("detection_class_entities", []),
        yxyx_boxes=detection_data.get("detection_boxes", []),
        scores=detection_data.get("detection_scores", []),
    )
    return record


def apply_detector(detector: Any, image_np: np.ndarray) -> ObjectRecord:
    try:
        image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
        detection_raw = detector(image_tf)
        record = get_record(detection_raw)
        return record
    except KeyboardInterrupt as e:
        logger.warning("Interrupted by user")
        raise e
    except Exception as e:
        raise RuntimeError(f"Failed to apply detector: {e}") from e


class OpenImagesExtractor(BaseObjectExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--detector-url",
            default="https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
            help="url of the detector to use",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.detector_url: str = args.detector_url

        if not self.gpu:
            tf.config.set_visible_devices([], "GPU")
            logger.warning("No GPU detected, using CPU for inference.")

        self.detector = hub.load(self.detector_url).signatures["default"]  # type: ignore
        logger.info(f"Loaded detector from {self.detector_url}")

    def extract_path(self, frame_path: Path) -> ObjectRecord:
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame path does not exist: {frame_path}")

        image_np = load_image_pil(frame_path)
        record = apply_detector(self.detector, image_np)
        record._id = frame_path.stem
        return record

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        records: list[ObjectRecord] = []
        for frame_path in frame_paths:
            try:
                record = self.extract_path(frame_path)
                records.append(record)
                logger.info(
                    f"Processed {record._id} with {len(record.labels or [])} labels"
                )
            except Exception as e:
                logger.error(f"Error processing {frame_path}: {e}")
        return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openimages extractor")
    OpenImagesExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = OpenImagesExtractor(args)
    extractor.run()
