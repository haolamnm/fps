import argparse
import collections
import concurrent.futures
import copy
import gzip
import itertools
import json
import math
import operator
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from ...common.files import FileJSONL
from ...common.types import ObjectItem, ObjectRecord
from ...common.utils import get_logger, load_config

logger = get_logger(__name__)


def get_labels_counter(record: ObjectRecord) -> collections.Counter:
    return collections.Counter(obj.label for obj in record.objects or [])


def load_hypersets(hypersets_path: Path) -> pd.Series:
    hypersets = pd.read_csv(hypersets_path, sep=";", usecols=[0, 1], names=["set", "hyperset"], index_col=0)
    hypersets = hypersets.squeeze("columns")  # Convert to Series
    assert isinstance(hypersets, pd.Series), (
        "Hypersets should be a pandas Series with set names as index and hypersets as values"
    )

    # Drop empty hypersets, split hypersets by comma, strip whitespace
    hypersets = hypersets.dropna().str.split(",").apply(lambda x: list(map(str.strip, x)))

    return hypersets


def process_record(record: ObjectRecord, config: dict[str, Any]) -> ObjectRecord:
    # Labels are usually a list of int that correspond to class indices
    # In this case, we use the word "labels" to refer to the class names
    labels = record.names or []
    scores = record.scores or []
    yxyx_boxes = record.yxyx_boxes or []

    # Parallel arrays to list of objects
    objects = (
        ObjectItem(detector=record.detector, label=label, score=score, yxyx_box=yxyx_box)
        for label, score, yxyx_box in zip(labels, scores, yxyx_boxes, strict=True)
    )

    # Filter by score threshold
    threshold: float = config.get("threshold", {}).get(record.detector, 0.0)
    objects = filter(lambda x: x.score >= threshold, objects)

    # Filter by normalized box area
    def _get_area(yxyx_box: tuple[float, float, float, float]) -> float:
        y0, x0, y1, x1 = yxyx_box
        return (y1 - y0) * (x1 - x0)

    min_area: float = config.get("min_area", 0.0)
    objects = filter(lambda x: _get_area(x.yxyx_box) >= min_area, objects)

    # Exclude labels (pre label ptraining)
    exclude_labels: list[str] = config.get("exclude_labels", {}).get(record.detector, [])
    objects = filter(lambda x: x.label not in exclude_labels, objects)

    # Normalize labels
    objects = (
        ObjectItem(
            detector=obj.detector,
            label=obj.label.lower().replace(" ", "_"),
            score=obj.score,
            yxyx_box=obj.yxyx_box,
        )
        for obj in objects
    )

    # Apply label mapping
    mapping: dict[str, str] = config.get("label_map", {}).get(record.detector, {})
    if mapping:
        objects = (
            ObjectItem(
                detector=obj.detector,
                label=mapping.get(obj.label, obj.label),
                score=obj.score,
                yxyx_box=obj.yxyx_box,
            )
            for obj in objects
        )

    # Exclude labels (post label ptraining)
    exclude_labels = config.get("exclude_labels_post", {}).get("all", [])
    objects = filter(lambda x: x.label not in exclude_labels, objects)

    record.objects = list(objects)
    return record


def process_input_file(input_file_path: Path, config: dict[str, Any]) -> Iterator[ObjectRecord]:
    with gzip.open(input_file_path, "rt") as file:
        records = map(str.rstrip, file)
        records = map(json.loads, records)
        records = (ObjectRecord(**x) for x in records)
        records = (process_record(x, config) for x in records)
        yield from records


def merge_records(records: Iterable[ObjectRecord]) -> ObjectRecord:
    if len({record._id for record in records}) != 1:
        raise ValueError("All records must have the same _id")

    objects = [record.objects or [] for record in records]
    for record in records:
        record.objects = None

    merged_objects: list[ObjectItem] = list(reduce(operator.add, objects))

    dict_records = [asdict(record) for record in records]
    merged_records = dict(collections.ChainMap(*dict_records))
    merged_records["objects"] = merged_objects

    return ObjectRecord(**merged_records)


def add_hypersets(record: ObjectRecord, hypersets: pd.Series) -> ObjectRecord:
    # Generate hyperset records for each object in the record
    def _generate_augumented_objects(objects: list[ObjectItem]) -> Iterator[ObjectItem]:
        for obj in objects:
            yield obj

            if obj.detector == "colors":
                continue

            hyperset_labels: list[str] = hypersets.get(obj.label, [])
            for hyperset_label in hyperset_labels:
                hyperset_obj = copy.deepcopy(obj)
                hyperset_obj.label = hyperset_label
                hyperset_obj.is_hyperset = True
                yield hyperset_obj

    # Substitute the original objects with the augmented ones
    objects = record.objects or []
    augumented_objects = _generate_augumented_objects(objects)
    record.objects = list(augumented_objects)
    return record


def _get_area(yxyx_box: tuple[float, float, float, float]) -> float:
    y0, x0, y1, x1 = yxyx_box
    return max(0, y1 - y0) * max(0, x1 - x0)


def _get_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    y0_1, x0_1, y1_1, x1_1 = box1
    y0_2, x0_2, y1_2, x1_2 = box2

    # Calculate intersection
    inter_y0 = max(y0_1, y0_2)
    inter_x0 = max(x0_1, x0_2)
    inter_y1 = min(y1_1, y1_2)
    inter_x1 = min(x1_1, x1_2)

    if inter_y0 >= inter_y1 or inter_x0 >= inter_x1:
        return 0.0  # No intersection

    intersection_area = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
    union_area = _get_area(box1) + _get_area(box2) - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def _get_nms(objects: Iterable[ObjectItem], iou_threshold: float) -> list[ObjectItem]:
    # Non-Maximum Suppression (NMS) to filter overlapping boxes
    objects = sorted(objects, key=lambda x: x.score, reverse=True)
    num_objects = len(objects)

    kept_objects: list[ObjectItem] = []
    for i in range(num_objects):
        box1 = objects[i].yxyx_box
        is_maximum = True
        for j in range(i + 1, num_objects):
            box2 = objects[j].yxyx_box
            if _get_iou(box1, box2) >= iou_threshold:
                is_maximum = False
                break

        if is_maximum:
            kept_objects.append(objects[i])

    return kept_objects


def apply_non_maximum_suppression(record: ObjectRecord, iou_threshold: float = 0.5) -> ObjectRecord:
    colors = [obj for obj in record.objects or [] if obj.detector == "colors"]
    objects = [obj for obj in record.objects or [] if obj.detector != "colors"]

    def _get_key(obj: ObjectItem) -> tuple[str, str]:
        return (obj.detector, obj.label)

    objects.sort(key=_get_key)
    objects = itertools.chain.from_iterable(
        _get_nms(group, iou_threshold) for _, group in itertools.groupby(objects, key=_get_key)
    )
    objects = list(objects)
    record.objects = objects + colors

    return record


def _str_encode_positional_boxes(
    objects: Iterable[ObjectItem],
    num_rows: int = 7,
    num_cols: int = 7,
    tolerance: float = 0.1,
) -> str:
    def _str_encode_object(object: ObjectItem) -> list[str]:
        y0, x0, y1, x1 = object.yxyx_box
        label = object.label

        x_tolerance = tolerance / num_cols
        y_tolerance = tolerance / num_rows

        start_col = math.floor((max(0, x0) + x_tolerance) * num_cols)
        start_row = math.floor((max(0, y0) + y_tolerance) * num_rows)
        end_col = math.floor((min(1, x1) - x_tolerance) * num_cols)
        end_row = math.floor((min(1, y1) - y_tolerance) * num_rows)

        surrogate_texts: list[str] = []
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                col = chr(ord("a") + col)  # Convert to column letter
                surrogate_texts.append(f"{row}{col}{label}")

        return surrogate_texts

    # Ignore hypersets in box encoding tokens
    objects = filter(lambda x: not x.is_hyperset, objects)
    tokens = map(_str_encode_object, objects)
    tokens = itertools.chain.from_iterable(tokens)
    tokens = sorted(tokens)  # Sort tokens to ensure consistent order

    surrogate_text = " ".join(tokens)
    return surrogate_text


def _str_encode_counts(objects: Iterable[ObjectItem], monochrome: float, thresholds: dict[str, float]) -> str:
    label_counts = collections.Counter()

    tokens: list[str] = []
    for obj in objects:
        key = (obj.label, obj.detector)
        label_counts[key] += 1
        label_count = label_counts[key]

        frequency = ""
        if obj.detector == "colors":
            if label_count > 1:
                continue
            label_count = ""
        elif not obj.is_hyperset:
            confidence = obj.score - thresholds.get(obj.detector, 0.0)
            frequency = int(10 * confidence / 2 + 2)
            frequency = f"|{frequency}"

        token = f"4wc{obj.label}{label_count}{frequency}"
        tokens.append(token)

    # Handle monochrome
    monochrome_token = "colorframe" if monochrome > thresholds.get("monochrome", 0.01) else "grayframe"
    monochrome_token = f"4wc{monochrome_token}"
    tokens.append(monochrome_token)

    tokens.sort()
    surrogate_text = " ".join(tokens)
    return surrogate_text


def str_encode(record: ObjectRecord, thresholds: dict[str, float]) -> ObjectRecord:
    objects = record.objects or []
    monochrome = record.monochrome or 0.0

    return ObjectRecord(
        _id=record._id,
        detector=record.detector,
        object_box_str=_str_encode_positional_boxes(objects),
        object_cnt_str=_str_encode_counts(objects, monochrome, thresholds),
    )


def get_objects_info(objects: Iterable[ObjectItem]) -> str:
    # For debugging purposes
    objects = filter(lambda x: not x.is_hyperset, objects)
    objects = filter(lambda x: x.detector != "colors", objects)

    detector_nicknames = {
        "mask_rcnn_lvis": "MASK",
        "vfnet_X-101-64x4d": "VFN64",
        "frcnn_incep_resnetv2_openimagesv4": "FRCNN",
    }

    label_counts = collections.Counter()
    tokens: list[str] = []
    for obj in objects:
        detector = detector_nicknames.get(obj.detector, obj.detector)
        key = (detector, obj.label)
        label_counts[key] += 1
        label_count = label_counts[key]

        token = f"{obj.label}{label_count}({detector}:{obj.score:.2f})"
        tokens.append(token)

    tokens.sort()
    surrogate_text = " ".join(tokens)
    return surrogate_text


def process_merged_record(
    record: ObjectRecord, config: dict[str, Any], hypersets: pd.Series
) -> tuple[ObjectRecord, collections.Counter]:
    # Add hypersets to the record
    record = add_hypersets(record, hypersets)

    # Perform Non-Maximum Suppression (NMS)
    record = apply_non_maximum_suppression(record)

    # Get objects info after NMS for debugging
    objects_after_nms = record.objects or []
    object_info = get_objects_info(objects_after_nms)

    # Get the labels counter
    labels_counter = get_labels_counter(record)

    # Build the STR encoded representation
    thresholds = config.get("threshold", {})
    record = str_encode(record, thresholds)

    record.object_info = object_info

    return record, labels_counter


def process_video_id(
    input_files: list[Path],
    str_output_file: Path,
    cnt_output_file: Path,
    config: dict[str, Any],
    hypersets: pd.Series,
    force: bool = False,
    flush_interval: int = 20,
    video_id: str = "",
):
    if not force and str_output_file.exists() and cnt_output_file.exists():
        logger.info(f"Skipping object encoding, using existing files: {str_output_file} and {cnt_output_file}")
        return

    assert all(input_file.exists() for input_file in input_files), "One or more input files do not exist"

    # Apply per-detector processing
    records = (process_input_file(x, config) for x in input_files)

    # Merge records of different detectors with the same _id
    merged_records = map(merge_records, zip(*records, strict=True))

    # Apply all processing steps (hypersets, NMS, counting, STR encoding)
    records_and_counters = (process_merged_record(x, config, hypersets) for x in merged_records)

    # If forced, delete old file
    if force and str_output_file.exists():
        str_output_file.unlink()

    object_counter = collections.Counter()

    str_output_file.parent.mkdir(parents=True, exist_ok=True)
    with FileJSONL(str(str_output_file), flush_interval) as file:
        for record, labels_counter in records_and_counters:
            file.save(record)
            object_counter += labels_counter
    logger.info(f"Processed {len(object_counter)} objects for video ID {video_id}")
    logger.info(f"Saved STR encoded objects to {str_output_file}")

    cnt_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cnt_output_file, "w") as file:
        json.dump(object_counter, file)
    logger.info(f"Saved object counts to {cnt_output_file}")


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)["index"]["objects"]
    hypersets = load_hypersets(args.hypersets_path)

    if not args.video_ids:
        process_video_id(
            args.input_file_templates,
            args.str_output_file_template,
            args.cnt_output_file_template,
            config=config,
            hypersets=hypersets,
            force=args.force,
            flush_interval=args.flush_interval,
        )

    logger.info("Starting parallel processing of video IDs")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = []
        for video_id in args.video_ids:
            input_files = [Path(str(template).format(video_id=video_id)) for template in args.input_file_templates]
            str_output_file = Path(str(args.str_output_file_template).format(video_id=video_id))
            cnt_output_file = Path(str(args.cnt_output_file_template).format(video_id=video_id))
            task = executor.submit(
                process_video_id,
                input_files,
                str_output_file,
                cnt_output_file,
                config,
                hypersets,
                args.force,
                args.flush_interval,
                video_id=video_id,
            )
            tasks.append(task)

        for furture in concurrent.futures.as_completed(tasks):
            try:
                furture.result()
            except Exception as e:
                logger.error(f"Error processing video ID: {e}")
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="str-objects encoder")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("~/fps/config.yaml").expanduser(),
        help="path to the configuration file",
    )
    parser.add_argument(
        "--hypersets-path",
        type=Path,
        default=Path("~/fps/hypersets.csv").expanduser(),
        help="path to the csv hypersets file",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=20,
        help="number of objects to process before flushing to disk",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="force re-encoding of objects even if they already exist",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        required=True,
        help="list of video IDs to process",
    )
    parser.add_argument(
        "str_output_file_template",
        type=Path,
        help="template for str-encoded output files, e.g. 'output/str-objects-{video_id}.jsonl.gz'",
    )
    parser.add_argument(
        "cnt_output_file_template",
        type=Path,
        help="template for output files with object counts, e.g. 'output/cnt-objects-{video_id}.jsonl.gz'",
    )
    parser.add_argument(
        "input_file_templates",
        type=Path,
        nargs="+",
        help="templates for input files with detected objects, {video_id} will be replaced",
    )

    args = parser.parse_args()
    main(args)
