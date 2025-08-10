import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import open_clip
import torch
import torch.nn.functional as F
import torch.utils
from PIL import Image

from ...common.extractors import BaseFrameExtractor
from ...common.types import FeatureRecord
from ...common.utils import get_logger

logger = get_logger(__name__)


class FrameListDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths: list[Path], processor: Any) -> None:
        self.frame_paths = frame_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Any:
        frame_path = self.frame_paths[index]
        image = Image.open(frame_path).convert("RGB")
        encoding = self.processor(image)
        return encoding


class OpenCLIPExtractor(BaseFrameExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            type=str,
            default="ViT-L-14",
            choices=["ViT-L-14", "ViT-B-32", "ViT-B-16"],
            help="name of the model to use for feature extraction",
        )
        parser.add_argument(
            "--pretrained",
            default="laion2b_s32b_b82k",
            type=str,
            choices=[
                "laion2b_s32b_b82k",
                "datacomp_xl_s13b_b90k",
            ],
            help="pretrained model to use for feature extraction",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="number of frames to process in each batch",
        )
        parser.add_argument(
            "--num-workers",
            default=8,
            type=int,
            help="number of worker threads for data loading",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.model_name: str = args.model_name
        self.pretrained: str = args.pretrained
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers

        # Ensure the cache directory exists
        cache_dir = "~/.cache/open_clip"
        os.makedirs(cache_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() and self.gpu else "cpu"
        self.model, self.processor, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
            cache_dir=cache_dir,
        )
        logger.info(f"Loaded model {self.pretrained} on device {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model = torch.compile(self.model)
        logger.info("Compiled model for performance optimization")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[FeatureRecord]:
        chunk_size = self.batch_size * 5

        current_chunk: list[Path] = []
        for frame_path in frame_paths:
            current_chunk.append(frame_path)

            if len(current_chunk) >= chunk_size:
                logger.info(f"Processing chunk of {len(current_chunk)} frames")
                yield from self._process_chunk(
                    current_chunk, self.batch_size, self.num_workers
                )
                current_chunk = []

        if current_chunk:
            logger.info(f"Processing chunk of {len(current_chunk)} frames")
            yield from self._process_chunk(
                current_chunk, self.batch_size, self.num_workers
            )

    def _process_chunk(
        self, frame_paths: list[Path], batch_size: int, num_workers: int
    ) -> Iterator[FeatureRecord]:
        dataset = FrameListDataset(frame_paths, self.processor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        with torch.no_grad():
            for frames in dataloader:
                frames = frames.to(self.device)
                features = self.model.encode_image(frames).float()  # type: ignore
                features = F.normalize(features, dim=-1, p=2)
                features = features.cpu().numpy()

                for frame_path, feature in zip(frame_paths, features):
                    yield FeatureRecord(
                        _id=frame_path.stem, feature_vector=feature.tolist()
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openclip extractor")
    OpenCLIPExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = OpenCLIPExtractor(args)
    extractor.run()
