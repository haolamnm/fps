import argparse
import csv
import itertools
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import more_itertools

from .files import FileHDF5, FileJSONL
from .types import FeatureRecord, Frame, ObjectRecord, Scene
from .utils import get_logger

logger = get_logger(__name__)


class BaseExtractor:
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=8,
            help="number of frames to process in each slice of the dataset",
        )
        parser.add_argument(
            "--force",
            default=False,
            action="store_true",
            help="force re-extraction even if the output file already exists",
        )
        parser.add_argument(
            "--gpu",
            default=False,
            action="store_true",
            help="use GPU for extraction if available",
        )
        parser.add_argument(
            "--flush-interval",
            type=int,
            default=20,
            help="flush the output file every N frames processed",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self.chunk_size: int = args.chunk_size
        self.force: bool = args.force
        self.gpu: bool = args.gpu
        self.flush_interval: int = args.flush_interval


class BaseFrameExtractor(BaseExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_dir",
            type=str,
            help="directory containing the frames to extract features from",
        )
        parser.add_argument(
            "-n",
            "--features-name",
            type=str,
            required=True,
            help="name of the feature to extract (used hdf5 metadata)",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="path to the output file where features will be saved (.h5)",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.frames_dir: Path = Path(args.frames_dir)
        self.features_name: str = args.features_name
        self.output_path: str = args.output

    def _load_frames(self) -> list[Frame]:
        if not self.frames_dir.is_dir():
            raise ValueError(f"Frames directory '{self.frames_dir}' does not exist or is not a directory")
        frames: list[Frame] = []
        frame_paths = sorted(self.frames_dir.glob("*.jpg")) or sorted(self.frames_dir.glob("*.png"))
        for frame_path in frame_paths:
            frame = Frame(video_id=self.frames_dir.name, _id=frame_path.stem, path=frame_path)
            frames.append(frame)

        if not frames:
            raise ValueError(f"No frames found in directory '{self.frames_dir}'")

        return frames

    def _get_output_file(self, video_id: str, read_only: bool = False) -> FileHDF5:
        self.output_path = self.output_path.format(video_id=video_id)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        return FileHDF5(
            file_path=self.output_path,
            features_name=self.features_name,
            read_only=read_only,
            flush_interval=self.flush_interval,
        )

    def extract_list(self, frame_paths: list[Path]) -> list[FeatureRecord]:
        raise NotImplementedError("This method should be implemented in subclasses")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[FeatureRecord]:
        assert self.chunk_size > 0, "Chunk size must be greater than 0"

        batched_frames = more_itertools.chunked(frame_paths, self.chunk_size)
        batched_records = map(self.extract_list, batched_frames)
        records = itertools.chain.from_iterable(batched_records)
        yield from records

    def _skip_extracted_frames(self, frames: list[Frame]) -> Iterator[Frame]:
        frame_groups = sorted(
            ((frame.video_id, frame._id, frame.path) for frame in frames),
            key=lambda x: x[0],
        )
        for video_id, group in itertools.groupby(frame_groups, key=lambda x: x[0]):
            with self._get_output_file(video_id, read_only=True) as file:
                skipping_frames: list[Frame] = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        skipping_frames.append(Frame(video_id=video_id, _id=frame_id, path=frame_path))
            logger.info(f"Skipping {len(frames) - len(skipping_frames)} frames for video '{video_id}'")
            yield from skipping_frames

    def run(self) -> None:
        frames = self._load_frames()

        if not self.force and Path(self.output_path).exists():
            frames = list(self._skip_extracted_frames(frames))

        if not frames:
            logger.info("Exiting because all frames are already extracted")
            return

        # Unpack frames into video IDs, frame IDs, and paths
        video_ids, frame_ids, frame_paths = zip(
            *[(frame.video_id, frame._id, frame.path) for frame in frames], strict=True
        )
        records = self.extract_iterable(frame_paths)

        # Group records by video ID
        triples = zip(video_ids, frame_ids, records, strict=True)
        video_groups: dict[str, list[tuple[str, FeatureRecord]]] = defaultdict(list)
        for video_id, frame_id, record in triples:
            video_groups[video_id].append((frame_id, record))

        num_videos = len(video_groups)
        num_records = sum(len(items) for items in video_groups.values())

        for video_id, items in video_groups.items():
            feature_records = [
                FeatureRecord(_id=frame_id, feature_vector=record.feature_vector) for frame_id, record in items
            ]
            with self._get_output_file(video_id, read_only=False) as file:
                file.save_all(feature_records, force=self.force)

        logger.info(f"Saved features to {Path(self.output_path)}")
        logger.info(f"Extracted {num_records} features from {num_videos} videos")


class BaseObjectExtractor(BaseExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_dir",
            type=str,
            help="path to the input frames directory",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="path to the output file where extracted objects will be saved (.jsonl.gz)",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.frames_dir: Path = Path(args.frames_dir)
        self.output_path: str = args.output

    def _load_frames(self) -> list[Frame]:
        if not self.frames_dir.is_dir():
            raise ValueError(f"Frames directory '{self.frames_dir}' does not exist or is not a directory")
        frames: list[Frame] = []
        frame_paths = sorted(self.frames_dir.glob("*.jpg")) or sorted(self.frames_dir.glob("*.png"))
        for frame_path in frame_paths:
            frame = Frame(video_id=self.frames_dir.name, _id=frame_path.stem, path=frame_path)
            frames.append(frame)

        if not frames:
            raise ValueError(f"No frames found in directory '{self.frames_dir}'")

        return frames

    def _get_output_file(self, video_id: str) -> FileJSONL:
        self.output_path = self.output_path.format(video_id=video_id)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        return FileJSONL(
            file_path=self.output_path,
            flush_interval=self.flush_interval,
        )

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        raise NotImplementedError("This method should be implemented in subclasses")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[ObjectRecord]:
        assert self.chunk_size > 0, "Chunk size must be greater than 0"

        batched_frames = more_itertools.chunked(frame_paths, self.chunk_size)
        batched_records = map(self.extract_list, batched_frames)
        records = itertools.chain.from_iterable(batched_records)
        yield from records

    def _skip_extracted_frames(self, frames: list[Frame]) -> Iterator[Frame]:
        frame_groups = sorted(
            ((frame.video_id, frame._id, frame.path) for frame in frames),
            key=lambda x: x[0],
        )
        for video_id, group in itertools.groupby(frame_groups, key=lambda x: x[0]):
            with self._get_output_file(video_id) as file:
                skipping_frames: list[Frame] = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        skipping_frames.append(Frame(video_id=video_id, _id=frame_id, path=frame_path))
            logger.info(f"Skipping {len(frames) - len(skipping_frames)} frames for video '{video_id}'")
            yield from skipping_frames

    def run(self) -> None:
        frames = self._load_frames()

        if not self.force and Path(self.output_path).exists():
            frames = list(self._skip_extracted_frames(frames))

        if not frames:
            logger.info("Exiting because all frames are already extracted")
            return

        # Unpack frames into video IDs, frame IDs, and paths
        video_ids, frame_ids, frame_paths = zip(
            *[(frame.video_id, frame._id, frame.path) for frame in frames], strict=True
        )
        records = self.extract_iterable(frame_paths)

        # Group records by video ID
        triples = zip(video_ids, frame_ids, records, strict=True)
        video_groups: dict[str, list[tuple[str, ObjectRecord]]] = defaultdict(list)
        for video_id, frame_id, record in triples:
            video_groups[video_id].append((frame_id, record))

        num_videos = len(video_groups)
        num_records = sum(len(items) for items in video_groups.values())

        for video_id, items in video_groups.items():
            object_records = [ObjectRecord(**{"_id": frame_id, **asdict(record)}) for frame_id, record in items]
            with self._get_output_file(video_id) as file:
                file.save_all(object_records, force=self.force)

        logger.info(f"Saved features to {Path(self.output_path)}")
        logger.info(f"Extracted {num_records} features from {num_videos} videos")


class BaseVideoExtractor(BaseExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_dir",
            type=str,
            help="path to the input frames directory",
        )
        parser.add_argument(
            "-n",
            "--features-name",
            type=str,
            required=True,
            help="name of the feature extractor to use",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="path to the output file where extracted frames will be saved (.h5)",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.frames_dir: Path = Path(args.frames_dir)
        self.features_name: str = args.features_name
        self.output_path: str = args.output

    def _load_frames(self) -> Iterator[Scene]:
        if not self.frames_dir.is_dir():
            raise ValueError(f"Frames directory '{self.frames_dir}' does not exist or is not a directory")

        frame_paths = sorted(self.frames_dir.glob("*.jpg")) or sorted(self.frames_dir.glob("*.png"))
        if not frame_paths:
            raise ValueError(f"No frames found in directory '{self.frames_dir}'")

        video_id = self.frames_dir.name
        frame_ids = [frame_path.stem for frame_path in frame_paths]
        frames = list(zip(itertools.repeat(video_id), frame_ids, frame_paths))

        # For each video, read the 'scenes.csv" file to get scene metadata
        for video_id, group in itertools.groupby(frames, key=lambda x: x[0]):
            frame_ids, frame_paths = zip(*((frame_id, frame_path) for _, frame_id, frame_path in group), strict=True)
            frame_ids = list(map(str, frame_ids))
            frame_paths = list(map(Path, frame_paths))

            scenes_file = frame_paths[0].parent / f"{video_id}-scenes.csv"
            if scenes_file.is_file():
                logger.info(f"Found scenes file {scenes_file} for video ID {video_id}")

            escaped_video_id = re.escape(video_id)
            candidates = (scenes_file.parents[2] / "videos").glob(f"{video_id}.*")
            video_paths = [
                candidate for candidate in candidates if re.match(rf"{escaped_video_id}\.[0-9a-zA-Z]+", candidate.name)
            ]
            if not video_paths:
                raise ValueError(f"Could not find video file for {video_id} in {scenes_file.parents[2] / 'videos'}")

            video_path = video_paths[0]
            logger.info(f"Found video file {video_path} for video ID {video_id}")

            # Read the scenes CSV file to get scene metadata
            with open(scenes_file) as file:
                reader = csv.DictReader(file)
                frame_id_to_metadata = {
                    int(row["Scene Number"]): (
                        int(row["Start Frame"]),
                        float(row["Start Time (seconds)"]),
                        int(row["End Frame"]),
                        float(row["End Time (seconds)"]),
                    )
                    for row in reader
                }

            for frame_id, frame_path in zip(frame_ids, frame_paths, strict=True):
                scene_id = int(re.split("[-_]", frame_path.stem)[-1])
                if scene_id not in frame_id_to_metadata:
                    logger.warning(f"Scene ID {scene_id} not found in {scenes_file}, skipping frame {frame_path}")
                    continue
                start_frame, start_time, end_frame, end_time = frame_id_to_metadata[scene_id]
                scene = Scene(
                    video_id=video_id,
                    _id=frame_id,
                    video_path=video_path,
                    start_frame=start_frame,
                    start_time=start_time,
                    end_frame=end_frame,
                    end_time=end_time,
                )
                yield scene

    def _get_output_file(self, video_id: str, read_only: bool = False) -> FileHDF5:
        self.output_path = self.output_path.format(video_id=video_id)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        return FileHDF5(
            file_path=self.output_path,
            features_name=self.features_name,
            read_only=read_only,
            flush_interval=self.flush_interval,
        )

    def extract_list(self, scenes: list[Scene]) -> list[FeatureRecord]:
        raise NotImplementedError("This method should be implemented in subclasses")

    def extract_iterable(self, scenes: Iterable[Scene]) -> Iterator[FeatureRecord]:
        assert self.chunk_size > 0, "Chunk size must be greater than 0"

        batched_scenes = more_itertools.chunked(scenes, self.chunk_size)
        batched_records = map(self.extract_list, batched_scenes)
        records = itertools.chain.from_iterable(batched_records)
        yield from records

    def _skip_extracted_scenes(self, scenes: list[Scene]) -> Iterator[Scene]:
        for video_id, group in itertools.groupby(scenes, key=lambda x: x.video_id):
            with self._get_output_file(video_id) as file:
                skipping_scenes: list[Scene] = []
                for scene in group:
                    if scene._id not in file:
                        skipping_scenes.append(scene)

            logger.info(f"Skipping {len(scenes) - len(skipping_scenes)} scenes for video '{video_id}'")
            yield from skipping_scenes

    def run(self) -> None:
        scenes = list(self._load_frames())

        if not self.force and Path(self.output_path).exists():
            scenes = list(self._skip_extracted_scenes(scenes))

        if not scenes:
            logger.info("Exiting because all scenes are already extracted")
            return

        records = self.extract_iterable(scenes)
        video_groups: dict[str, list[tuple[str, FeatureRecord]]] = defaultdict(list)
        for scene, record in zip(scenes, records, strict=True):
            video_groups[scene.video_id].append((scene._id, record))

        num_videos = len(video_groups)
        num_records = sum(len(items) for items in video_groups.values())

        for video_id, items in video_groups.items():
            feature_records = [
                FeatureRecord(_id=scene_id, feature_vector=record.feature_vector) for scene_id, record in items
            ]
            with self._get_output_file(video_id, read_only=False) as file:
                file.save_all(feature_records, force=self.force)

        logger.info(f"Saved features to {Path(self.output_path)}")
        logger.info(f"Extracted {num_records} features from {num_videos} videos")
