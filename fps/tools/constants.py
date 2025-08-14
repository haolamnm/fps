SUPPORTED_VIDEO_FORMATS = [
    ".webm",
    ".mpg",
    ".mp2",
    ".mpeg",
    ".mpe",
    ".mpv.ogg",
    ".mp4",
    ".m4p",
    ".m4v",
    ".avi",
    ".wmv",
    ".mov",
    ".qt",
    ".flv",
    ".swf.h264",
    ".3g2",
    ".3gp",
    ".m4v",
]

DEFAULT_DETECTION_PARAMS = [
    "detect-adaptive",
    "detect-threshold",
]

SUCCESS_EXIT_CODE = 0
FAILURE_EXIT_CODE = 1

# Constant fields in the scenes CSV
SCENE_NUMBER = "Scene Number"
START_FRAME = "Start Frame"
START_TIMECODE = "Start Timecode"
START_SECONDS = "Start Time (seconds)"
END_FRAME = "End Frame"
END_TIMECODE = "End Timecode"
END_SECONDS = "End Time (seconds)"
LENGTH_FRAMES = "Length (frames)"
LENGTH_TIMECODE = "Length (timecode)"
LENGTH_SECONDS = "Length (seconds)"

# Service name mappings
SERVICE_MAPPING = {
    "clip-openai": "clip",
    "clip-laion": "openclip",
    "clip-datacomp": "openclip",
    "mrcnn-lvis": "mmdet",
    "vfnet64-coco": "mmdet",
    "frcnn-oiv4": "openimages",
}

# Port mapping for services
PORT_MAPPING = {
    # Main port
    "main": 8080,
    # Index services
    "faiss-index-manager": 8090,
    # Analysis services
    "features-clip": 8081,
    "features-clip2video": 8082,
    "features-clipvip": 8083,
    "features-dinov2": 8084,
    "features-openclip": 8085,
}
