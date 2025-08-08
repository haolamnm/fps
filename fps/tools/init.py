from pathlib import Path
import shutil
import argparse


def get_skel_dir() -> Path:
    skel_dir = Path(__file__).parents[1] / "skel"
    if not skel_dir.exists():
        raise FileNotFoundError(f"Skeleton directory not found: {skel_dir}")
    return skel_dir


def copy_skel_to(dest: Path) -> None:
    skel_dir = get_skel_dir()
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skel_dir, dest, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="init")
    parser.add_argument(
        "dest",
        type=Path,
        nargs="?",
        default=Path.home() / "fps",
        help="destination directory for skeleton files",
    )
    args = parser.parse_args()

    try:
        copy_skel_to(args.dest)
        print(f"[init] Skeleton files copied to {args.dest}")
    except FileNotFoundError as e:
        print(f"[init] Error: {e}")
