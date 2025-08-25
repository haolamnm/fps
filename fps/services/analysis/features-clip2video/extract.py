import argparse
import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision

from ...common.extractors import BaseVideoExtractor
from ...common.types import FeatureRecord, Scene
from ...common.utils import get_logger
from .config import Config
from .utils import get_device, load_model, set_seed

logger = get_logger(__name__)


def get_video_duration(video_path: Path, num_ffprobe_threads: int = 2) -> float:
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


@dataclass
class SceneConfig:
    min_scene_duration: float = 0.0
    frames_per_second: int = 5
    max_frames: int = 100
    frame_sampling_rate: int = 2
    frame_size: int = 224
    frame_transform: torchvision.transforms.Compose | None = None
    num_ffmpeg_threads: int = 2


def load_scene(
    scene: Scene,
    scene_config: SceneConfig,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    video = torch.empty(
        1,
        scene_config.max_frames,
        scene_config.frame_size,
        scene_config.frame_size,
        3,
        dtype=torch.uint8,
    )
    video_mask = torch.zeros(1, scene_config.max_frames, dtype=torch.long)

    scene_duration = scene.end_time - scene.start_time
    num_frames = scene.end_frame - scene.start_frame

    if scene_duration == 0 or num_frames <= 1:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds and {num_frames} frames"
            "Expanding scene to 0.5 seconds duration."
        )
        scene.start_time = max(0, scene.start_time - 0.5 / 2)
        scene.end_time = scene.start_time + 0.5
        scene_duration = scene.end_time - scene.start_time

    if scene_duration < scene_config.min_scene_duration:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds, which is less than the minimum "
            f"{scene_config.min_scene_duration} seconds. Padding scene to minimum duration."
        )
        video_duration = get_video_duration(scene.video_path, scene_config.num_ffmpeg_threads)
        padding = (scene_config.min_scene_duration - scene_duration) / 2
        scene.start_time = max(0, scene.start_time - padding)
        scene.end_time = min(video_duration, scene.end_time + scene_config.min_scene_duration)
        scene_duration = scene.end_time - scene.start_time

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "fatal",
        "-threads",
        f"{scene_config.num_ffmpeg_threads}",
        "-ss",
        f"{scene.start_time:.2f}",
        "-i",
        str(scene.video_path),
        "-t",
        f"{scene_duration:.2f}",
        "-r",
        f"{scene_config.frames_per_second}",
        "-q",
        "0",
        "-vf",
        "scale=320x240",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:",
    ]
    ffmpeg = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    video_bytes, _ = ffmpeg.communicate()
    if ffmpeg.returncode != 0:
        logger.error(f"Error ffmpeg failed to process video {scene.video_path}")
        return video, video_mask, scene._id

    try:
        video = torch.frombuffer(video_bytes, dtype=torch.uint8).reshape(-1, 240, 320, 3).detach().clone()
    except Exception as e:
        logger.error(f"Failed to reshape video bytes for scene {scene._id}: {e}")
        return video, video_mask, scene._id

    video = video.permute(0, 3, 1, 2)
    video = video[:: scene_config.frame_sampling_rate, ...]
    video = video / 255.0

    if scene_config.frame_transform is not None:
        try:
            video = scene_config.frame_transform(video)
        except Exception as e:
            logger.error(f"Error frame transform failed for scene {scene._id}: {e}")
            return video, video_mask, scene._id

    video_frames = video.shape[0]
    if video_frames > scene_config.max_frames:
        index = np.linspace(0, video_frames - 1, scene_config.max_frames).astype(int)
        video = video[index, ...]
        video_frames = scene_config.max_frames

    else:
        padding = torch.zeros(
            scene_config.max_frames - video_frames,
            3,
            scene_config.frame_size,
            scene_config.frame_size,
        )
        video = torch.cat([video, padding], dim=0)

    video = video.unsqueeze(1)
    video = video.unsqueeze(0)
    video_mask[0, :video_frames] = True

    return video, video_mask, scene._id


class CLIP2VideoListDataset(torch.utils.data.Dataset):
    def __init__(self, scenes: list[Scene], scene_config: SceneConfig) -> None:
        self.scenes = scenes
        self.scene_config = scene_config

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        scene = self.scenes[index]
        video, video_mask, scene_id = load_scene(scene, self.scene_config)
        return video, video_mask, scene_id


class CLIP2VideoExtractor(BaseVideoExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--min-scene-duration",
            type=float,
            default=3.0,
            help="minimum duration of a scene to extract in seconds",
        )
        parser.add_argument(
            "--fps",
            type=float,
            default=5,
            help="fps to sample frames from the video",
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
        parser.add_argument(
            "--num-threads",
            type=int,
            default=2,
            help="number of threads to use for each ffmpeg worker",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers

        self.config = Config(
            checkpoint_dir="checkpoint",
            clip_path="checkpoint/ViT-B-32.pt",
        )
        self.config.gpu = self.gpu and torch.cuda.is_available()
        set_seed(self.config)

        self.scene_config = SceneConfig(
            min_scene_duration=args.min_scene_duration,
            frames_per_second=args.fps,
            max_frames=self.config.max_frames,
            frame_sampling_rate=self.config.feature_framerate,
            frame_size=args.input_size,
            frame_transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        args.input_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                    ),
                    torchvision.transforms.CenterCrop(args.input_size),
                    torchvision.transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            ),
            num_ffmpeg_threads=args.num_threads,
        )
        self.device, self.num_gpu = get_device(
            self.config,
            local_rank=self.config.local_rank,
        )
        self.model = load_model(self.config, self.device)
        self.model.eval()
        logger.info(f"Loaded model at {self.config.checkpoint_dir} on device {self.device.type}")

    def forward_batch(self, batch: tuple[torch.Tensor, torch.Tensor, list[str]]) -> list[FeatureRecord]:
        video, video_mask, scene_ids = batch
        video, video_mask = video.to(self.device), video_mask.to(self.device)

        visual_output = self.model.get_visual_output(video, video_mask)
        video_features = self.model.get_video_features(visual_output, video_mask)

        records: list[FeatureRecord] = []
        for index, scene_id in enumerate(scene_ids):
            feature = video_features[index].cpu().numpy().tolist()
            record = FeatureRecord(_id=scene_id, feature_vector=feature)
            records.append(record)

        return records

    def extract_list(self, scenes: list[Scene]) -> list[FeatureRecord]:
        dataset = CLIP2VideoListDataset(scenes, self.scene_config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        with torch.no_grad():
            raw_records: list[list[FeatureRecord]] = []
            for batch in dataloader:
                logger.info(f"Processing batch of size {len(batch[0])}")
                batch_records = self.forward_batch(batch)
                raw_records.append(batch_records)

        records = list(itertools.chain.from_iterable(raw_records))
        return records

    def extract_iterable(self, scenes: argparse.Iterable[Scene]) -> itertools.Iterator[FeatureRecord]:
        scenes_list = list(scenes)
        dataset = CLIP2VideoListDataset(scenes_list, self.scene_config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        with torch.no_grad():
            for batch in dataloader:
                logger.info(f"Processing batch of size {len(batch[0])}")
                batch_records = self.forward_batch(batch)
                yield from batch_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip2video extractor")
    CLIP2VideoExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = CLIP2VideoExtractor(args)
    extractor.run()
