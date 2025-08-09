import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ...common.utils import load_config


class FaissIndexHandler:
    def __init__(self, index: Any, ids: list[str]) -> None:
        self.index = index
        self.ids = ids
        self.idmap = {_id: i for i, _id in enumerate(ids)}

    def search(
        self, feature_vector: np.ndarray, k: int
    ) -> tuple[list[str], list[float]]:
        distances, indices = self.index.search(feature_vector, k)  # noqa: E741

        # If no results are found
        if indices.size == 0 or indices[0][0] == -1:
            raise HTTPException(status_code=404, detail="No results found")

        # Take the first result in the batch
        indices = indices[0]
        distances = distances[0]

        # Filter out -1 results
        valid = indices >= 0
        indices = indices[valid]
        distances = distances[valid]

        ids = [(self.ids[i]) for i in indices]
        return ids, distances.tolist()


loaded_indices: dict[str, FaissIndexHandler] = {}


def load_index(features_name: str) -> FaissIndexHandler:
    if features_name in loaded_indices:
        return loaded_indices[features_name]

    index_path = Path(f"~/fps/data/faiss-index_{features_name}.faiss").expanduser()
    idmap_path = Path(f"~/fps/data/faiss-idmap_{features_name}.txt").expanduser()
    if not index_path.exists() or not idmap_path.exists():
        raise FileNotFoundError(f"Index or ID map not found for {features_name}")

    index = faiss.read_index(str(index_path))
    with open(idmap_path, "r") as file:
        ids = [line.strip() for line in file.readlines()]

    index_handler = FaissIndexHandler(index, ids)
    loaded_indices[features_name] = index_handler
    return index_handler


class SearchRequest(BaseModel):
    feature_vector: list[float]
    features_name: str
    k: int = 10


def create_app() -> FastAPI:
    app = FastAPI(title="faiss index handler")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/search")
    def search(request: SearchRequest):
        if not request.features_name:
            raise HTTPException(status_code=400, detail="features_name is required")

        if not request.feature_vector:
            raise HTTPException(status_code=400, detail="feature_vector is required")

        if request.features_name not in loaded_indices:
            try:
                load_index(request.features_name)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))

        if request.k <= 0:
            raise HTTPException(status_code=400, detail="k must be a positive integer")

        # Make sure the feature is a 2D numpy array
        feature_vector = np.atleast_2d(request.feature_vector).astype(np.float32)
        index_handler = loaded_indices[request.features_name]

        # Perform the search
        frame_ids, scores = index_handler.search(feature_vector, request.k)
        results = [
            {"id": frame_id, "score": score}
            for frame_id, score in zip(frame_ids, scores)
            if score >= 0.10  # Filter out low scores
        ]

        return {"results": results}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faiss index handler")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="host to run the server on",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="port to run the server on",
    )
    parser.add_argument(
        "--lazy",
        default=False,
        action="store_true",
        help="only load indices when first requested",
    )
    parser.add_argument(
        "--config-path",
        default=Path("~/fps/config.yaml").expanduser(),
        type=Path,
        help="path to the configuration file",
    )
    args = parser.parse_args()

    if not args.config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {args.config_path}")

    # When lazy loading is false, load all indices at startup
    if not args.lazy:
        config = load_config(args.config_path)

        # Nested dictionary
        available_features = config["analysis"]["features"]
        indexed_features = config["index"]["features"]

        features_names: list[str] = [
            key
            for key, value in indexed_features.items()
            if value["index_engine"] == "faiss" and key in available_features
        ]
        for features_name in features_names:
            try:
                load_index(features_name)
            except FileNotFoundError as e:
                print(f"Error loading index for {features_name}: {e}")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
