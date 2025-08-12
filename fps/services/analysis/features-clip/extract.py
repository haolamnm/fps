import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import transformers
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

    def __getitem__(self, idx: int) -> Any:
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert("RGB")
        encoding = self.processor(images=[image], return_tensors="pt")
        return encoding

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {key: torch.concat([item[key] for item in batch]) for key in batch[0]}


class CLIPExtractor(BaseFrameExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            type=str,
            default="openai/clip-vit-large-patch14",
            choices=[
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-large-patch14-336",
            ],
            help="name of the CLIP model to use for feature extraction",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="number of frames to process in each batch",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="number of worker threads for data loading",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.model_name: str = args.model_name
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers

        self.device = "cuda" if torch.cuda.is_available() and self.gpu else "cpu"
        self.processor = transformers.CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = transformers.CLIPModel.from_pretrained(self.model_name).to(
            self.device  # type: ignore
        )
        logger.info(f"Loaded model {self.model_name} on device {self.device}")
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
                yield from self._process_chunk(current_chunk, self.batch_size, self.num_workers)
                current_chunk = []

        if current_chunk:
            logger.info(f"Processing chunk of {len(current_chunk)} frames")
            yield from self._process_chunk(current_chunk, self.batch_size, self.num_workers)

    def _process_chunk(self, frame_paths: list[Path], batch_size: int, num_workers: int) -> Iterator[FeatureRecord]:
        dataset = FrameListDataset(frame_paths, self.processor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=FrameListDataset.collate_fn,
        )
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model.get_image_features(**inputs)  # type: ignore
                features = torch.nn.functional.normalize(outputs, dim=-1, p=2).cpu().numpy()

                for frame_path, feature in zip(frame_paths, features, strict=False):
                    yield FeatureRecord(_id=frame_path.stem, feature_vector=feature.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip extractor")
    CLIPExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = CLIPExtractor(args)
    extractor.run()
