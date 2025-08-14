import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any

from ..services.common.utils import get_logger, load_config
from .constants import SUCCESS_EXIT_CODE
from .utils import run_command

logger = get_logger(__name__)

mapping = {
    "clip-openai": "clip",
    "clip-laion": "openclip",
    "clip-datacomp": "openclip",
    "mrcnn-lvis": "mmdet",
    "vfnet64-coco": "mmdet",
    "frcnn-oiv4": "openimages",
}


def parse_cmd_params(params: dict[str, Any] | None) -> list[str]:
    cmd_params: list[str] = []
    if params is None:
        return cmd_params

    for key, value in params.items():
        cmd_params.extend([f"--{key}", f"{value}"])
    return cmd_params


def extract_features(
    extractor: str,
    video_id: str,
    collection_dir: Path,
    force: bool,
    gpu: bool,
    params: dict[str, Any] | None = None,
) -> int:
    output_template = collection_dir / f"features-{extractor}" / "{video_id}" / f"{{video_id}}-{extractor}.h5"
    frames_dir = collection_dir / "selected-frames" / video_id

    cmd_params = parse_cmd_params(params)
    service = f"fps.services.analysis.features-{mapping.get(extractor, extractor)}.extract"
    command = list(
        itertools.chain(
            [sys.executable, "-m", service, str(frames_dir)],
            ["--force"] if force else [],
            ["--gpu"] if gpu else [],
            cmd_params,
            ["-n", extractor, "-o", str(output_template)],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "extracting features")


def detect_objects(
    detector: str, video_id: str, collection_dir: Path, force: bool, gpu: bool, params: dict[str, Any] | None = None
) -> int:
    output_template = (
        collection_dir / f"objects-{detector}" / "{video_id}" / f"{{video_id}}-objects-{detector}.jsonl.gz"
    )
    frames_dir = collection_dir / "selected-frames" / video_id

    cmd_params = parse_cmd_params(params)
    service = f"fps.services.analysis.objects-{mapping.get(detector, detector)}.extract"
    command = list(
        itertools.chain(
            [sys.executable, "-m", service, str(frames_dir)],
            ["--force"] if force else [],
            ["--gpu"] if gpu else [],
            cmd_params,
            ["-o", str(output_template)],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "detecting objects")


def cluster_frames(features: str, video_id: str, collection_dir: Path, force: bool) -> int:
    output_path = collection_dir / "cluster-codes" / video_id / f"{video_id}-cluster-codes.jsonl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and output_path.exists():
        logger.info(f"Cluster codes for {video_id} already exist at {output_path}. Skipping.")
        return SUCCESS_EXIT_CODE

    features_file = collection_dir / f"features-{features}" / video_id / f"{video_id}-{features}.h5"
    assert features_file.exists(), f"Features file {features_file} does not exist."

    service = "fps.services.analysis.frames-cluster.extract"
    command = list(
        itertools.chain(
            [sys.executable, "-m", service, "-i", str(features_file)],
            ["--force"] if force else [],
            ["-o", str(output_path)],
            ["--video-id", video_id],
        )
    )
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "clustering frames")


def analyze_videos(
    video_ids: list[str], analyzers: list[str], collection_dir: Path, replace: bool, gpu: bool, config: dict[str, Any]
) -> None:
    analysis_config = config.get("analysis", {})
    logger.info(f"Analyzing videos: {video_ids}")

    for video_id in video_ids:
        active_detectors = analysis_config.get("objects", {})
        active_extractors = analysis_config.get("features", {})
        clustering_features = analysis_config.get("frames_cluster", {}).get("feature", None)

        # Keep only given analyzers
        if analyzers:
            available = (
                list(active_detectors.keys())
                + list(active_extractors.keys())
                + (["frames-cluster"] if clustering_features else [])
            )
            assert all(analyzer in available for analyzer in analyzers), f"Invalid analyzers: {', '.join(analyzers)}"
            active_detectors = {k: v for k, v in active_detectors.items() if k in analyzers}
            active_extractors = {k: v for k, v in active_extractors.items() if k in analyzers}
            clustering_features = clustering_features if "frames-cluster" in analyzers else None

        for detector, params in active_detectors.items():
            logger.info(f"Detecting objects with {detector} for video {video_id}")
            detect_objects(detector, video_id, collection_dir, replace, gpu, params=params)

        for extractor, params in active_extractors.items():
            logger.info(f"Extracting features with {extractor} for video {video_id}")
            extract_features(extractor, video_id, collection_dir, replace, gpu, params=params)

        if clustering_features:
            logger.info(f"Clustering frames with {clustering_features} for video {video_id}")
            cluster_frames(clustering_features, video_id, collection_dir, replace)

        logger.info(f"Analysis completed for video {video_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyze")
    parser.add_argument(
        "--id",
        dest="video_ids",
        nargs="+",
        default=None,
        help="video ids to analyze (default: all imported videos)",
    )
    parser.add_argument(
        "--replace",
        default=False,
        action="store_true",
        help="replace existing features (default: False)",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="use GPU for feature extraction (default: False)",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="enable verbose logging for debug (default: False)",
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path.home() / "fps",
        help="path to the collection directory (default: ~/fps)",
    )
    parser.add_argument(
        "analyzers",
        nargs="*",
        default=None,
        help="analyzers to run (default: all available analyzers)",
    )

    args = parser.parse_args()

    collection_dir: Path = args.collection_path.expanduser().resolve()
    frames_dir: Path = collection_dir / "selected-frames"
    video_ids: list[str] = args.video_ids or [video_id.name for video_id in frames_dir.iterdir() if video_id.is_dir()]
    analyzers: list[str] = args.analyzers or []
    replace: bool = args.replace
    gpu: bool = args.gpu
    verbose: bool = args.verbose
    config: dict[str, Any] = load_config(str(collection_dir / "config.yaml"))

    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)

    analyze_videos(video_ids, analyzers, collection_dir, replace, gpu, config)
