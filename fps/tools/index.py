import argparse
import collections
import csv
import gzip
import itertools
import json
import logging
import sys
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..services.common.utils import get_logger, load_config
from .constants import (
    END_FRAME,
    END_SECONDS,
    START_FRAME,
    START_SECONDS,
    SUCCESS_EXIT_CODE,
)
from .utils import run_command

logger = get_logger(__name__)


def str_encode_objects(video_id: str, collection_dir: Path, config: dict[str, Any], force: bool = False) -> int:
    str_objects_template = collection_dir / "str-objects" / "{video_id}" / "{video_id}-str-objects.jsonl.gz"
    cnt_objects_template = collection_dir / "cnt-objects" / "{video_id}" / "{video_id}-cnt-objects.json"

    detectors = config.get("analysis", {}).get("objects", [])
    objects_templates = [
        collection_dir / f"objects-{detector}" / "{video_id}" / f"{{video_id}}-objects-{detector}.jsonl.gz"
        for detector in detectors
    ]
    objects_templates = sorted(map(str, objects_templates))

    command = list(
        itertools.chain(
            [sys.executable, "-m", "fps.services.index.str-object-encoder.encode"],
            ["--config-path", str(collection_dir / "config.yaml")],
            ["--force"] if force else [],
            [str(str_objects_template), str(cnt_objects_template)],
            objects_templates,
            ["--video-ids", video_id],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "encoding STR objects")


def str_encode_features(video_id: str, collection_dir: Path, features_name: str, force: bool = False) -> int:
    input_template = collection_dir / f"features-{features_name}" / "{video_id}" / f"{{video_id}}-{features_name}.h5"
    str_encoder_file = collection_dir / f"str-features-encoder-{features_name}.pkl"
    str_features_template = (
        collection_dir / "str-features" / "{video_id}" / f"{{video_id}}-str-features-{features_name}.jsonl.gz"
    )

    command = list(
        itertools.chain(
            [sys.executable, "-m", "fps.services.index.str-feature-encoder.encode"],
            ["--config-path", str(collection_dir / "config.yaml")],
            ["--force"] if force else [],
            [str(input_template), str(str_encoder_file), str(str_features_template)],
            ["--video-ids", video_id],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "encoding STR features")


def prepare_lucene_doc(video_id: str, collection_dir: Path, force: bool = False) -> int:
    lucene_documents_dir = collection_dir / "lucene-documents" / video_id
    lucene_documents_dir.mkdir(parents=True, exist_ok=True)
    lucene_documents_file = lucene_documents_dir / f"{video_id}-lucene-docs.jsonl.gz"

    scenes_file = collection_dir / "selected-frames" / video_id / f"{video_id}-scenes.csv"

    if not force and lucene_documents_file.exists():
        logger.info(f"Lucene documents for {video_id} already exist at {lucene_documents_file}. Skipping.")
        return SUCCESS_EXIT_CODE

    def map_scenes(csv_path: Path) -> Iterator[dict[str, Any]]:
        with open(csv_path, newline="") as file:
            for row in csv.DictReader(file):
                yield {
                    "start_frame": int(row[START_FRAME]),
                    "end_frame": int(row[END_FRAME]),
                    "middle_frame": (int(row[START_FRAME]) + int(row[END_FRAME])) // 2,
                    "start_time": float(row[START_SECONDS]),
                    "end_time": float(row[END_SECONDS]),
                    "middle_time": (float(row[START_SECONDS]) + float(row[END_SECONDS])) / 2,
                }

    scene_docs = map_scenes(scenes_file)

    # Prepare objects fields of records
    str_objects_file = collection_dir / "str-objects" / video_id / f"{video_id}-str-objects.jsonl.gz"

    def map_objects(jsonl_path: Path) -> Iterator[dict[str, Any]]:
        with gzip.open(jsonl_path, "rt") as records:
            records = map(str.rstrip, records)
            records = map(json.loads, records)
            yield from records

    str_object_docs = map_objects(str_objects_file) if str_objects_file.exists() else itertools.repeat({})

    # Prepare frames cluster codes
    clusters_file = collection_dir / "cluster-codes" / video_id / f"{video_id}-cluster-codes.jsonl.gz"
    clusters_docs = map_objects(clusters_file) if clusters_file.exists() else itertools.repeat({})

    # Prepare features fields of records
    str_features_dir = collection_dir / "str-features" / video_id
    str_features_files = sorted(str_features_dir.glob(f"{video_id}-str-features-*.jsonl.gz"))

    def map_features(jsonl_path: Path) -> Iterator[dict[str, Any]]:
        features_name = jsonl_path.name[len(f"{video_id}-str-features-") : -len(".jsonl.gz")]
        with gzip.open(jsonl_path, "rt") as records:
            records = map(str.rstrip, records)
            records = map(json.loads, records)

            def rename(record: dict[str, Any]) -> dict[str, Any]:
                record[f"features_{features_name}_str"] = record.pop("feature_str", "")
                return record

            records = map(rename, records)
            yield from records

    str_feature_docs = map(map_features, str_features_files)

    # Merge fields into a single document
    str_documents = [scene_docs, str_object_docs, clusters_docs]
    if str_features_files:
        str_documents.extend(str_feature_docs)

    def merge_documents(docs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for records in zip(*docs, strict=True):
            merged_record = dict(collections.ChainMap(*records))
            yield merged_record

    records = merge_documents(str_documents)

    def fix_fieldnames(record: dict[str, Any]) -> dict[str, Any]:
        # ID field
        _id = record.pop("_id", "")
        record["image_id"] = _id
        record["video_id"] = video_id
        record["collection"] = "fps"

        # TODO: Change to use a more consistent naming scheme
        record["txt"] = record.pop("object_box_str", "")
        record["objects"] = record.pop("object_cnt_str", "")
        record["objects_info"] = record.pop("object_info", "")

        if "features_dinov2_str" in record:
            record["features"] = record.pop("features_dinov2_str", "")

        record["aladin"] = record.pop("features_aladin_str", "")
        return record

    records = map(fix_fieldnames, records)

    # Save merged records to a JSONL file
    with gzip.open(lucene_documents_file, "wt") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved Lucene documents for {video_id} to {lucene_documents_file}")
    return SUCCESS_EXIT_CODE


def add_to_lucene_index(video_id: str, collection_dir: Path, force: bool = False) -> int:
    documents_template = collection_dir / "lucene-documents" / "{video_id}" / "{video_id}-lucene-docs.jsonl.gz"
    lucene_index_dir = collection_dir / "lucene-index"

    command = list(
        itertools.chain(
            ["java", "-jar", "lucene-index-manager.jar", str(lucene_index_dir), "add"],
            ["--force"] if force else [],
            [str(documents_template), "--video-ids", video_id],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "adding to lucene index")


def add_to_faiss_index(features_name: str, video_ids: list[str] | None, force: bool = False) -> int:
    faiss_index_file = collection_dir / f"faiss-index_{features_name}.faiss"
    faiss_idmap_file = collection_dir / f"faiss-idmap_{features_name}.txt"

    features_dir = collection_dir / f"features-{features_name}"
    features_input = [features_dir]

    bulk_mode = video_ids is None
    assert bulk_mode or len(video_ids) > 0, "video_ids must be a non-empty list"

    if not bulk_mode:
        features_input = [features_dir / video_id / f"{video_id}-{features_name}.h5" for video_id in video_ids]

    service = "fps.services.index.faiss-index-manager.build"
    command = list(
        itertools.chain(
            [sys.executable, "-m", service, str(faiss_index_file), str(faiss_idmap_file)],
            ["--config-path", str(collection_dir / "config.yaml")],
            ["create"] if bulk_mode else ["add"],
            ["--force"] if force else [],
            [str(f) for f in features_input],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, None, "adding to faiss index")


def count_objects(force: bool = False) -> int:
    cnt_objects_dir = collection_dir / "cnt-objects"
    cnt_objects_dir.mkdir(parents=True, exist_ok=True)
    cnt_objects_file = collection_dir / "objects_doc_freq.csv"

    if not force and cnt_objects_file.exists():
        logger.info(f"{cnt_objects_file} already exist. Skipping.")
        return SUCCESS_EXIT_CODE

    # Count objects
    count = collections.Counter()
    for cnt_file in cnt_objects_dir.glob("**/*-cnt-objects.json"):
        with cnt_file.open("r") as file:
            count += collections.Counter(json.load(file))

    # Save to CSV in alphabetical order
    with cnt_objects_file.open("w") as file:
        writer = csv.writer(file)
        for key in sorted(count.keys()):
            writer.writerow([key, count[key]])

    return SUCCESS_EXIT_CODE


def index_videos(video_ids: list[str], config: dict[str, Any], replace: bool = False, phases: list[str] | None = None):
    index_config: dict[str, Any] = config.get("index", {})

    str_objects: dict[str, Any] = index_config.get("objects", {})
    indexed_features: dict[str, Any] = index_config.get("features", {})

    str_features = [k for k, v in indexed_features.items() if v["index_engine"] == "str"]
    faiss_features = [k for k, v in indexed_features.items() if v["index_engine"] == "faiss"]

    update_lucene = any([str_objects, str_features])
    objects_count = len(config.get("analysis", {}).get("objects", {})) > 0

    # Keep only requested phases
    if phases:
        str_objects = str_objects if "objects" in phases else {}
        str_features = [f for f in str_features if f in phases]
        faiss_features = [f for f in faiss_features if f in phases]
        update_lucene = "lucene" in phases
        objects_count = "objects-count" in phases

    threads: list[threading.Thread] = []

    for video_id in video_ids:
        if str_objects:
            thread = threading.Thread(
                target=str_encode_objects,
                kwargs={"video_id": video_id, "collection_dir": collection_dir, "config": config, "force": replace},
            )
            thread.start()
            threads.append(thread)

        for features_name in str_features:
            logger.info(f"Encoding STR features for {video_id} with {features_name}")
            str_encode_features(video_id, collection_dir, features_name, force=replace)

        if update_lucene:
            for thread in threads:
                thread.join()

            logger.info(f"Preparing Lucene documents for {video_id}")
            prepare_lucene_doc(video_id, collection_dir, force=replace)

            logger.info(f"Adding {video_id} to Lucene index")
            add_to_lucene_index(video_id, collection_dir, force=replace)

        for features_name in faiss_features:
            logger.info(f"Adding {video_id} to FAISS index for {features_name}")
            add_to_faiss_index(features_name, [video_id], force=replace)

        for thread in threads:
            thread.join()

    if objects_count:
        logger.info("Counting objects across all videos")
        count_objects(force=replace)

    if not threads:
        logger.info("No indexing tasks were started")

    logger.info("Indexing completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index")
    parser.add_argument(
        "--id",
        dest="video_ids",
        nargs="+",
        default=None,
        help="video ids to index (default: all analyzed videos)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="replace existing index files (default: False)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="enable verbose output (default: False)",
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path.home() / "fps",
        help="path to the collection directory (default: ~/fps)",
    )
    parser.add_argument(
        "phases",
        nargs="*",
        default=None,
        help="indexing phases to run (default: all phases)",
    )
    args = parser.parse_args()

    collection_dir: Path = args.collection_path.expanduser().resolve()
    video_ids: list[str] = args.video_ids or [
        video_id.name for video_id in (collection_dir / "selected-frames").iterdir() if video_id.is_dir()
    ]
    phases: list[str] = args.phases or []
    replace: bool = args.replace
    verbose: bool = args.verbose
    config: dict[str, Any] = load_config(str(collection_dir / "config.yaml"))

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    index_videos(video_ids, config, replace=replace, phases=phases)
    logger.info("Indexing process completed successfully")
