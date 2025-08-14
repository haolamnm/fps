import argparse
import logging
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda

from ..services.common.utils import get_logger
from .constants import (
    DEFAULT_DETECTION_PARAMS,
    END_FRAME,
    END_SECONDS,
    END_TIMECODE,
    LENGTH_FRAMES,
    LENGTH_SECONDS,
    LENGTH_TIMECODE,
    SCENE_NUMBER,
    START_FRAME,
    START_SECONDS,
    START_TIMECODE,
    SUCCESS_EXIT_CODE,
    SUPPORTED_VIDEO_FORMATS,
)
from .utils import run_command

logger = get_logger(__name__)


@np.vectorize
def seconds_to_timecode(seconds: float) -> str:
    """
    Convert seconds to timecode string (HH:MM:SS.MS).
    """
    seconds = np.round(seconds, 3)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def post_process_scenes(scenes_file: Path, max_length: float) -> None:
    """
    Post-process scenes to split them into smaller segments if they exceed max_length.
    """

    # Read the scenes CSV file
    if not scenes_file.exists():
        raise FileNotFoundError(f"Scenes file {scenes_file} does not exist")
    scenes_df = pd.read_csv(scenes_file)

    fps_estimate: float = scenes_df.iloc[-1][END_FRAME] / scenes_df.iloc[-1][END_SECONDS]
    splitted_scenes = []

    # For each scene, check if it exceeds max_length and split if necessary
    for _, scene in scenes_df.iterrows():
        scene_number = scene[SCENE_NUMBER]
        scene_start_frame = scene[START_FRAME]
        scene_start_timecode = scene[START_TIMECODE]
        scene_start_seconds = scene[START_SECONDS]

        scene_end_frame = scene[END_FRAME]
        scene_end_timecode = scene[END_TIMECODE]
        scene_end_seconds = scene[END_SECONDS]

        scene_len_frames = scene[LENGTH_FRAMES]
        scene_len_timecode = scene[LENGTH_TIMECODE]
        scene_len_seconds = scene[LENGTH_SECONDS]

        # If the scene length is less than or equal to max_length, keep it as is
        if scene_len_seconds <= max_length:
            splitted_scenes.append(
                {
                    SCENE_NUMBER: scene_number,
                    START_FRAME: scene_start_frame,
                    START_TIMECODE: scene_start_timecode,
                    START_SECONDS: scene_start_seconds,
                    END_FRAME: scene_end_frame,
                    END_TIMECODE: scene_end_timecode,
                    END_SECONDS: scene_end_seconds,
                    LENGTH_FRAMES: scene_len_frames,
                    LENGTH_TIMECODE: scene_len_timecode,
                    LENGTH_SECONDS: scene_len_seconds,
                }
            )
            continue

        # Otherwise, split the scene into smaller segments
        num_splits = np.ceil(scene_len_seconds / max_length).astype(int)
        split_frames = np.linspace(scene_start_frame, scene_end_frame, num_splits + 1).astype(int)
        split_times = (split_frames - 1) / fps_estimate
        split_timecode = seconds_to_timecode(split_times)
        split_lengths_seconds = np.diff(split_times)
        split_lengths_frames = np.diff(split_frames)
        split_lengths_timecode = seconds_to_timecode(split_lengths_seconds)

        subscenes = pd.DataFrame(
            {
                SCENE_NUMBER: scene_number + np.linspace(0, 1, num_splits + 1)[:-1],
                START_FRAME: split_frames[:-1],
                START_TIMECODE: split_timecode[:-1],
                START_SECONDS: split_times[:-1],
                END_FRAME: split_frames[1:],
                END_TIMECODE: split_timecode[1:],
                END_SECONDS: split_times[1:],
                LENGTH_FRAMES: split_lengths_frames,
                LENGTH_TIMECODE: split_lengths_timecode,
                LENGTH_SECONDS: split_lengths_seconds,
            }
        ).to_dict(orient="records")

        splitted_scenes.extend(subscenes)

    # Create a new DataFrame for the splitted scenes
    splitted_scenes_df = pd.DataFrame(splitted_scenes)
    splitted_scenes_df[SCENE_NUMBER] = splitted_scenes_df.index + 1
    splitted_scenes_df.to_csv(scenes_file)


def copy_video(
    source_path: Path,
    video_id: str,
    collection_dir: Path,
    force: bool = False,
) -> tuple[str, Path]:
    """
    Copy a video file to the collection directory.
    """
    # Get video ID and target path
    video_id = video_id or source_path.stem
    video_ext = source_path.suffix.lower()
    target_path = collection_dir / "videos" / f"{video_id}{video_ext}"

    # Ensure the target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the target video already exists
    if target_path.exists() and (not force or target_path.samefile(source_path)):
        logger.debug(f"Skipping video {video_id}, already exists at {target_path}")
        return video_id, target_path

    # Otherwise, copy the video file
    shutil.copy2(source_path, target_path)
    return video_id, target_path


def build_resize_command(input_path: Path, tiny_output: Path, medium_output: Path, gpu: bool = False) -> list[str]:
    """
    Function to build the ffmpeg command for resizing videos.
    """
    use_gpu = torch.cuda.is_available() and gpu
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "-",
        "-nostats",
    ]

    if use_gpu:
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    command += ["-i", str(input_path)]

    # Output 1: Tiny
    command += (
        [
            "-vf",
            "scale_npp=146:-1:force_divisible_by=2",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-tune",
            "hq",
        ]
        if use_gpu
        else [
            "-vf",
            "scale=146:-1:force_divisible_by=2,pad='iw+mod(iw\\,2)':'ih+mod(ih\\,2)'",
            "-c:v",
            "libx264",
            "-preset",
            "slower",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
        ]
    )
    command += ["-c:a", "aac", "-b:a", "128k", str(tiny_output)]

    # Output 2: Medium
    command += (
        [
            "-vf",
            "scale_npp=-1:480:force_divisible_by=2",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-tune",
            "hq",
        ]
        if use_gpu
        else [
            "-vf",
            "scale=-1:480:force_divisible_by=2,pad='iw+mod(iw\\,2)':'ih+mod(ih\\,2)'",
            "-c:v",
            "libx264",
            "-preset",
            "slower",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
        ]
    )
    command += ["-c:a", "aac", "-b:a", "128k", str(medium_output)]

    return command


def create_resized_videos(
    video_path: Path, video_id: str, collection_dir: Path, force: bool = False, gpu: bool = False
) -> int:
    """
    Create resized versions of the video.
    """
    # Get the resized video directory
    resized_videos_dir = collection_dir / "resized-videos"
    tiny_video_file = resized_videos_dir / "tiny" / f"{video_id}-tiny.mp4"
    medium_video_file = resized_videos_dir / "medium" / f"{video_id}-medium.mp4"

    # Ensure directories exist
    resized_videos_dir.mkdir(parents=True, exist_ok=True)
    tiny_video_file.parent.mkdir(parents=True, exist_ok=True)
    medium_video_file.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing resized videos
    if not force and tiny_video_file.exists() and medium_video_file.exists():
        logger.debug(f"Skipping video {video_id}, already exist at {tiny_video_file} and {medium_video_file}")
        return SUCCESS_EXIT_CODE

    # Otherwise, create resized videos
    command = build_resize_command(video_path, tiny_video_file, medium_video_file, gpu=gpu)
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "creating resized videos")


def detect_scenes(
    video_path: Path,
    video_id: str,
    collection_dir: Path,
    scene_detection_params: list[str],
    scene_max_length: float,
    force: bool = False,
) -> int:
    """
    Detect scenes in the video.
    """
    # Get the selected frames directory and scene CSV file path
    selected_frames_dir = collection_dir / "selected-frames" / video_id
    scene_csv_file = selected_frames_dir / f"{video_id}-scenes.csv"

    # Ensure the selected frames directory exists
    selected_frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if scenes have already been detected
    if not force and scene_csv_file.exists():
        logger.debug(f"Skipping video {video_id}, already exists at {scene_csv_file}")
        return SUCCESS_EXIT_CODE

    # Otherwise, run scene detection
    command = (
        [
            "scenedetect",
            "--quiet",
            "--config",
            str(collection_dir / "scenedetect.cfg"),
            "--input",
            str(video_path),
            "--output",
            str(selected_frames_dir),
        ]
        + scene_detection_params
        + ["list-scenes", "--filename", f"{video_id}-scenes"]
    )
    logger.debug(f"Running command: {' '.join(command)}")
    run_command(command, video_id, "detecting scenes")

    # Post-process detected scences
    if scene_max_length > 0:
        post_process_scenes(scene_csv_file, scene_max_length)
        logger.debug(f"Post-processed scenes for {video_id} with max length {scene_max_length} seconds")

    return SUCCESS_EXIT_CODE


def extract_frames(video_path: Path, video_id: str, collection_dir: Path, force: bool = False) -> int:
    """
    Extract frames from the video.
    """
    # Get the selected frames directory and scenes CSV file path
    selected_frames_dir = collection_dir / "selected-frames" / video_id
    scenes_csv_file = selected_frames_dir / f"{video_id}-scenes.csv"

    # Get all PNG files in the selected frames directory
    selected_frames_dir.mkdir(parents=True, exist_ok=True)
    selected_frames_files = sorted(selected_frames_dir.glob("*.png"))

    def can_skip() -> bool:
        """
        Check if the scenes CSV file matches the number of selected frames.
        """
        if not scenes_csv_file.exists():
            return False
        with open(scenes_csv_file) as file:
            return len(file.readlines()) - 1 == len(selected_frames_files)

    # Check if frames have already been extracted
    if not force and scenes_csv_file.exists() and can_skip():
        logger.debug(f"Skipping video {video_id}, already exists in {selected_frames_dir / '*.png'}")
        return SUCCESS_EXIT_CODE

    # Otherwise, extract frames using scenedetect
    command = [
        "scenedetect",
        "--quiet",
        "--config",
        str(collection_dir / "scenedetect.cfg"),
        "--input",
        str(video_path),
        "--output",
        str(selected_frames_dir),
        "load-scenes",
        "--input",
        str(scenes_csv_file),
        "save-images",
        "--filename",
        f"{video_id}-$SCENE_NUMBER",
    ]
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "extracting frames")


def create_thumbnails(video_id: str, collection_dir: Path, force: bool = False) -> int:
    """
    Create thumbnails for the video.
    """
    # Get the selected frames and thumbnails directories
    selected_frames_dir = collection_dir / "selected-frames" / video_id
    thumbnails_dir = collection_dir / "thumbnails" / video_id

    # Ensure the directories exist
    selected_frames_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Get all selected frames and thumbnail files
    selected_frames_files = sorted(selected_frames_dir.glob("*.png"))
    thumbnail_files = sorted(thumbnails_dir.glob("*.jpg"))

    # Check if thumbnails already exist
    if not force and [f.stem for f in thumbnail_files] == [f.stem for f in selected_frames_files]:
        logger.debug(f"Skipping video {video_id}, already exists in {thumbnails_dir / '*.jpg'}")
        return SUCCESS_EXIT_CODE

    if not selected_frames_files:
        raise FileNotFoundError(f"No selected frames found for {video_id} in {selected_frames_dir}")

    num_digits = len(selected_frames_files[0].stem.split("-")[-1])

    input_pattern = selected_frames_dir / f"{video_id}-%0{num_digits}d.png"
    ouput_pattern = thumbnails_dir / f"{video_id}-%0{num_digits}d.jpg"

    # Otherwise, create thumbnails using ffmpeg
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "-",
        "-nostats",
        "-i",
        str(input_pattern),
        "-vf",
        "scale=192:-1",
        str(ouput_pattern),
    ]
    logger.debug(f"Running command: {' '.join(command)}")
    return run_command(command, video_id, "creating thumbnails")


def import_video(
    source_path: Path,
    video_id: str,
    collection_dir: Path,
    scene_detection_params: list[str],
    scene_max_length: float,
    use_all_paths: bool,
    replace: bool = False,
    gpu: bool = False,
) -> int:
    """
    Import a single video into the collection.
    """
    # Copy video file to the collection directory
    overwrite = replace if not use_all_paths else False
    video_id, target_path = copy_video(source_path, video_id, collection_dir, overwrite)
    logger.debug(f"Copied video {video_id} to {target_path}")

    # Create resized video files in background process
    thread = threading.Thread(
        target=create_resized_videos,
        args=(target_path, video_id, collection_dir, replace, gpu),
    )
    thread.start()

    # Detect scenes in the video
    detect_scenes(target_path, video_id, collection_dir, scene_detection_params, scene_max_length, replace)
    logger.debug(f"Detected scenes for {video_id} in {target_path}")

    # Extract frames
    extract_frames(target_path, video_id, collection_dir, replace)
    logger.debug(f"Extracted frames for {video_id} from {target_path}")

    # Create thumbnails
    create_thumbnails(video_id, collection_dir, replace)
    logger.debug(f"Created thumbnails for {video_id} in {collection_dir}")

    thread.join()
    logger.debug(f"Resized videos for {video_id} created in background thread")

    logger.info(f"Imported video {video_id} successfully")
    return SUCCESS_EXIT_CODE


def str_to_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="import")
    parser.add_argument(
        "video_path",
        type=Path,
        help="path to the video file to import",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="video ID to use for the imported video (default: filename without extension)",
    )
    parser.add_argument(
        "--replace",
        default=False,
        action="store_true",
        help="replace any existing video with the same ID (default: False)",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="use GPU for video processing (default: False, uses CPU)",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="enable verbose logging for debug (default: False)",
    )
    parser.add_argument(
        "--scene-detection-params",
        type=lambda v: str_to_list(v) if isinstance(v, str) else v,
        default=DEFAULT_DETECTION_PARAMS,
        help="scene detection parameters (comma-separated list, default: detect-adaptive,detect-threshold)",
    )
    parser.add_argument(
        "--scene-max-length",
        type=float,
        default=0.0,
        help="maximum length of scenes in seconds (default: 0.0, no splitting)",
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path.home() / "fps",
        help="path to the collection directory (default: ~/fps)",
    )
    args = parser.parse_args()

    # Define argument variables
    video_path: Path = args.video_path.expanduser().resolve()
    video_id: str = args.id.strip() or video_path.stem
    collection_dir: Path = args.collection_path.expanduser().resolve()
    replace: bool = args.replace
    verbose: bool = args.verbose
    gpu: bool = args.gpu
    scene_detection_params: list[str] = args.scene_detection_params
    scene_max_length: float = args.scene_max_length

    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)

    # Use all paths is specify a directory and no ID is given
    assert not (video_id and video_path is None), "Cannot specify ID without a video path"
    use_all_paths = video_path.is_dir() and not video_id

    if use_all_paths:
        logger.info(f"Importing all videos from directory: {video_path}")
        video_dir = collection_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_paths = sorted(
            video
            for video in video_dir.glob("*")
            if video.is_file() and video.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        )
        # Since we handle multiple video extensions, we need to ensure unique filenames
        assert len({v.stem for v in video_paths}) == len(video_paths), (
            "Found duplicate video filenames in the collection directory"
        )

    else:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Importing video: {video_path} with ID: {video_id}")
        video_paths = [video_path] if video_path.is_file() else []

    # Call import_video for each video path
    for video_path in video_paths:
        import_video(
            video_path,
            video_id,
            collection_dir,
            scene_detection_params,
            scene_max_length,
            use_all_paths,
            replace,
            gpu,
        )
