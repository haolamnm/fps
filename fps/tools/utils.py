import subprocess

from ..services.common.utils import get_logger
from .constants import FAILURE_EXIT_CODE, SUCCESS_EXIT_CODE

logger = get_logger(__name__)


def run_command(command: list[str], video_id: str | None, description: str = "running command") -> int:
    try:
        process = subprocess.run(
            command,
            text=True,
            check=True,
            capture_output=True,
            bufsize=1,
        )
        for line in process.stdout:
            line = line.strip()
            print(line)

        return SUCCESS_EXIT_CODE
    except subprocess.CalledProcessError as e:
        logger.error(f"Error {description} for video {video_id}: {e.stderr.strip()}")
        return e.returncode
    except Exception as e:
        logger.error(f"Unexpected error {description} for video {video_id}: {str(e)}")
        return FAILURE_EXIT_CODE
