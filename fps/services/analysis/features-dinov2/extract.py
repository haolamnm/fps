import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ...common.extractors import BaseFrameExtractor
from ...common.types import FeatureRecord
from ...common.utils import get_logger

warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available*")

logger = get_logger(__name__)


def load_image(image_path: Path, transform: T.Compose | None = None) -> Any:
    try:
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path.name}: {e}") from e


class FrameListDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths: list[Path]) -> None:
        self.frame_paths = frame_paths
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Any:
        frame_path = self.frame_paths[index]
        frame = load_image(frame_path, self.transform)
        return frame


class DinoV2Extractor(BaseFrameExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            default="dinov2_vits14",
            choices=(
                "dinov2_vits14",
                "dinov2_vitb14",
                "dinov2_vitl14",
                "dinov2_vitg14",
            ),
            help="name of the DINOv2 model to use",
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
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers

        self.device = "cuda" if torch.cuda.is_available() and self.gpu else "cpu"
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True,
            trust_repo=True,
        ).to(self.device)  # type: ignore
        self.model.eval()
        logger.info(f"Loaded model {self.model_name} on device {self.device}")

    def extract_list(self, frame_paths: list[Path]) -> list[FeatureRecord]:
        dataset = FrameListDataset(frame_paths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
        )
        features = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)
                feature = self.model(batch).cpu().numpy()
                features.append(feature)

        features = np.concatenate(features, axis=0).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        records = [
            FeatureRecord(_id=frame_path.stem, feature_vector=feature.tolist())
            for frame_path, feature in zip(frame_paths, features)
        ]
        return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dinov2 extractor")
    DinoV2Extractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = DinoV2Extractor(args)
    extractor.run()
