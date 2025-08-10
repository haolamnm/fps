import subprocess
from typing import Any

from ..services.common.utils import get_logger
from .constants import FAILURE_EXIT_CODE, SUCCESS_EXIT_CODE

logger = get_logger(__name__)


def run_command(
    command: list[str], video_id: str, description: str = "running command", env: dict[str, Any] | None = None
) -> int:
    try:
        subprocess.run(
            command,
            env=env,
            text=True,
            check=True,
            capture_output=True,
        )
        return SUCCESS_EXIT_CODE
    except subprocess.CalledProcessError as e:
        logger.error(f"Error {description} for video {video_id}: {e.stderr.strip()}")
        return e.returncode
    except Exception as e:
        logger.error(f"Unexpected error {description} for video {video_id}: {str(e)}")
        return FAILURE_EXIT_CODE
