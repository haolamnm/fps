import os
import random

import numpy as np
import torch

from ...common.utils import get_logger
from .config import Config
from .model import CLIP2VideoModel

logger = get_logger(__name__)


def set_seed(config: Config) -> None:
    """Set random seed for reproducibility."""
    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Config, local_rank: int = 0) -> tuple[torch.device, int]:
    """Load the device based on the configuration."""
    use_gpu = config.gpu and torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_gpu else torch.device("cpu")
    num_gpu = torch.cuda.device_count() if use_gpu else 0

    config.num_gpu = num_gpu
    return device, num_gpu


def load_model(config: Config, device: torch.device) -> CLIP2VideoModel:
    """Load the CLIP2Video model based on the configuration."""
    model_path = config.checkpoint_dir / f"pytorch_model.bin.{config.model_num}"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    state_dict = torch.load(model_path, map_location=device)
    model = CLIP2VideoModel.from_pretrained(
        cross_model_name=config.cross_model_name,
        cache_dir=config.cache_dir,
        state_dict=state_dict,
        task_config=config,
    ).to(device)

    return model
