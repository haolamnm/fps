import argparse
from collections.abc import Iterator
import gc
import math
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from av.container.input import InputContainer
from easydict import EasyDict
from transformers import AutoProcessor
from transformers.models.clip.configuration_clip import (
    CLIPConfig,
)

from ...common.extractors import BaseVideoExtractor
from ...common.types import FeatureRecord, Scene
from ...common.utils import get_logger
from .clipvip.clipvip.CLIP_VIP import CLIPModel

logger = get_logger(__name__)


def read_video_pyav(container: InputContainer, indices: np.ndarray, start_time: float, total_frames: int) -> np.ndarray:
    frames: list[av.VideoFrame] = []
    start_time_tb = int(start_time * av.time_base)
    container.seek(start_time_tb, any_frame=True)

    for index, frame in enumerate(container.decode(video=0)):
        if index > total_frames:
            break
        if index in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(num_frames: int, total_frames: int) -> np.ndarray:
    start_index = 0
    end_index = total_frames
    indices = np.linspace(start_index, end_index, num=num_frames)
    indices = np.clip(indices, start_index, end_index - 1).astype(np.int64)
    return indices


def get_video_duration(video_path: Path, num_ffprobe_threads: int = 2) -> float:
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-threads",
        f"{num_ffprobe_threads}",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video duration for {video_path}: {e}")
        return float("inf")


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split("\n"))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video dimensions for {video_path}: {e}")
        return 0, 0


def load_scene(scene: Scene, min_scene_duration: float) -> np.ndarray:
    num_frames = 12

    scene_width, scene_height = get_video_dimensions(scene.video_path)
    video = np.zeros((num_frames, scene_height, scene_width, 3), dtype=np.uint8)

    scene_duration = scene.end_time - scene.start_time

    if scene_duration < min_scene_duration:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds, which is less than the minimum "
            f"{min_scene_duration} seconds. Padding scene to minimum duration."
        )
        video_duration = get_video_duration(scene.video_path, num_ffprobe_threads=2)
        padding = (min_scene_duration - scene_duration) / 2
        scene.start_time = max(0, scene.start_time - padding)
        scene.end_time = min(video_duration, scene.end_time + min_scene_duration)
        scene_duration = scene.end_time - scene.start_time

    with av.open(scene.video_path.as_posix(), mode="r", metadata_errors="ignore") as container:
        video_stream = container.streams.video[0]

        if video_stream.duration is not None and video_stream.time_base is not None:
            video_duration = float(video_stream.duration * video_stream.time_base)
            if video_duration - scene.start_time < 3:
                logger.warning(f"Scene {scene._id} has less than 3 seconds of video left, using the last 3 seconds.")
                scene.start_time = video_duration - 3

        fps = video_stream.average_rate or video_stream.guessed_rate or 25
        total_frames = math.ceil(scene_duration * fps)

        if total_frames == 0:
            logger.warning(f"Scene {scene._id} has no frames, using 1 frame.")
            total_frames = 1

        indices = sample_frame_indices(num_frames, total_frames)
        try:
            video = read_video_pyav(container, indices, scene.start_time, total_frames)
        except Exception as e:
            logger.error(f"Error reading video for scene {scene._id} at {scene.video_path}: {e}")
    gc.collect()
    return video


class CLIPVIPListDataset(torch.utils.data.Dataset):
    def __init__(self, scenes: list[Scene], min_scene_duration: float) -> None:
        self.scenes = scenes
        self.min_scene_duration = min_scene_duration

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> np.ndarray:
        scene = self.scenes[index]
        return load_scene(scene, self.min_scene_duration)


class VideoCollate:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch):
        batch = [list(b) for b in batch]
        batch = self.processor(videos=batch, return_tensors="pt").pixel_values
        return batch


class CLIPVIPExtractor(BaseVideoExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            type=str,
            default="openai/clip-vit-base-patch16",
            choices=[
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
            ],
            help="name of the CLIP model to use for feature extraction",
        )
        parser.add_argument(
            "--min-scene-duration",
            type=float,
            default=3.0,
            help="minimum duration of a scene in seconds",
        )
        parser.add_argument(
            "--input-size",
            type=int,
            default=224,
            help="size of the input images to the model",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="batch size for processing",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="number of workers for data loading",
        )
        super().add_arguments(parser)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.model_name: str = args.model_name
        self.min_scene_duration: float = args.min_scene_duration
        self.input_size: int = args.input_size
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers

        self.device = "cuda" if torch.cuda.is_available() and self.gpu else "cpu"
        extra_config = EasyDict(
            {
                "type": "ViP",
                "temporal_size": 12,
                "if_use_temporal_embed": 1,
                "logit_scale_init_value": 4.60,
                "add_cls_num": 3,
            }
        )
        config = CLIPConfig.from_pretrained(self.model_name)
        config.vision_additional_config = extra_config

        raw_dict = torch.load(
            Path(__file__).parent / "checkpoint/pretrain_clipvip_base_16.pt",
            map_location=self.device,
        )
        clean_dict = {key.replace("clipmodel.", ""): value for key, value in raw_dict.items()}
        self.model = CLIPModel(config=config)  # type: ignore
        self.model.load_state_dict(clean_dict, strict=False)

        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()

        model_type = self.model_name.split("clip-vit-")[-1]
        self.processor = AutoProcessor.from_pretrained(f"microsoft/xclip-{model_type}")
        logger.info(f"Loaded model {self.model_name} on device {self.device}")

    def forward_batch(self, video: torch.Tensor) -> list[FeatureRecord]:
        video = video.to(self.device)
        inputs = {"if_norm": True, "pixel_values": video}

        features = self.model.get_image_features(**inputs)

        records = [
            FeatureRecord(_id="", feature_vector=feature.tolist()) for feature in features.detach().cpu().numpy()
        ]
        return records

    def extract_list(self, scenes: list[Scene]) -> list[FeatureRecord]:
        dataset = CLIPVIPListDataset(scenes, self.min_scene_duration)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=VideoCollate(self.processor),
        )

        records: list[FeatureRecord] = []

        with torch.no_grad():
            for batch in dataloader:
                batch_records = self.forward_batch(batch)
                records.extend(batch_records)

        for index, scene in enumerate(scenes):
            if index < len(records):
                records[index]._id = scene._id

        return records

    def extract_iterable(self, scenes: argparse.Iterable[Scene]) -> Iterator[FeatureRecord]:
        collate_fn = VideoCollate(self.processor)
        scenes_list = list(scenes)
        dataset = CLIPVIPListDataset(scenes_list, self.min_scene_duration)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            for batch in dataloader:
                batch_records = self.forward_batch(batch)
                yield from batch_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clipvip extractor")
    CLIPVIPExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = CLIPVIPExtractor(args)
    extractor.run()
