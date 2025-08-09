import argparse
import itertools
import multiprocessing
import os
from pathlib import Path
from queue import Empty
from typing import Any

import mmcv
from .mmdetection.mmdet.apis import inference_detector, init_detector
import numpy as np
import torch

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord


def convert_detections_to_record(
    detections: Any, detector: str, classes: list[str], image_hw: tuple[int, int], _id: str
) -> ObjectRecord:
    if detector == "mrcnn-lvis":
        boxes_and_scores = detections[0]
    elif detector in ["vfnet32-coco", "vfnet64-coco"]:
        boxes_and_scores = detections
    else:
        raise ValueError(f"Unknown detector: {detector}")

    num_instances_per_class = map(len, boxes_and_scores)
    labels = [[cls] * num for cls, num in zip(classes, num_instances_per_class)]
    labels = list(itertools.chain.from_iterable(labels))

    boxes_and_scores = np.concatenate(boxes_and_scores, axis=0)
    boxes_and_scores = boxes_and_scores[:, [1, 0, 3, 2, 4]]  # xyxy -> yxyx

    # Normalize coordinates and clip them in [0, 1]
    boxes = boxes_and_scores[:, :4] / np.tile(image_hw, 2)
    boxes = np.clip(boxes, 0, 1)
    scores = boxes_and_scores[:, 4]

    return ObjectRecord(
        _id=_id,
        scores=scores.tolist(),
        yxyx_boxes=boxes.tolist(),
        names=labels,
        detector=detector,
    )


class MMDetExtractor(BaseObjectExtractor):
    CONFIG_DIR = Path(__file__).parent / "mmdetection" / "configs"
    CHECKPOINT_DIR = Path(__file__).parent / "mmdetection" / "checkpoint"

    DETECTORS = {
        "vfnet32-coco": {
            "config": CONFIG_DIR
            / "vfnet"
            / "vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint": CHECKPOINT_DIR
            / "vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth",
        },
        "vfnet64-coco": {
            "config": CONFIG_DIR
            / "vfnet"
            / "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
            "checkpoint": CHECKPOINT_DIR
            / "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth",
        },
        "mrcnn-lvis": {
            "config": CONFIG_DIR
            / "lvis"
            / "mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py",
            "checkpoint": CHECKPOINT_DIR
            / "mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth",
        },
    }

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--detector",
            type=str,
            default="vfnet32-coco",
            choices=list(cls.DETECTORS.keys()),
            help="detector to use",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.detector: str = args.detector
        self.device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"

        config_file = str(self.DETECTORS[self.detector]["config"])
        checkpoint_file = str(self.DETECTORS[self.detector]["checkpoint"])
        self.model = init_detector(config_file, checkpoint_file, device=self.device)

    def extract_path(self, frame_path: Path) -> ObjectRecord:
        with torch.no_grad():
            image = mmcv.imread(str(frame_path))
            image_hw = image.shape[:2]
            detections = inference_detector(self.model, image)
            return convert_detections_to_record(
                detections, self.detector, self.model.CLASSES, image_hw, frame_path.name # type: ignore
            )

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        records = map(self.extract_path, frame_paths)
        return list(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmdet extractor")
    MMDetExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = MMDetExtractor(args)
    extractor.run()
