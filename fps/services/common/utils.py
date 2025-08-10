import logging

import yaml


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s : %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(name)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config
