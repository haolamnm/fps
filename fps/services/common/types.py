from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureRecord:
    _id: str
    feature_vector: list[float]


@dataclass
class ObjectItem:
    detector: str
    label: str
    score: float
    yxyx_box: tuple[float, float, float, float]
    is_hyperset: bool = False


@dataclass
class ObjectRecord:
    _id: str
    detector: str

    # Object detection fields
    scores: list[float] | None = None
    yxyx_boxes: list[tuple[float, float, float, float]] | None = None
    labels: list[int] | None = None
    entities: list[str] | None = None
    names: list[str] | None = None
    objects: list[ObjectItem] | None = None

    # Color-related fields
    monochrome: float | None = None

    # Frame clustering
    cluster_code: str | None = None

    # STR-related fields
    feature_str: str | None = None
    object_box_str: str | None = None
    object_cnt_str: str | None = None
    object_info: str | None = None


@dataclass
class Frame:
    video_id: str
    _id: str
    path: Path


@dataclass
class Scene:
    video_id: str
    _id: str
    video_path: Path
    start_frame: int
    start_time: float
    end_frame: int
    end_time: float
