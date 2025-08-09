import argparse
from pathlib import Path
from typing import Any

import numpy as np
import surrogate
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

loaded_encoders: dict[str, Any] = {}


def load_encoder(features_name: str) -> Any:
    if features_name in loaded_encoders:
        return loaded_encoders[features_name]

    encoder_path = Path(f"~/fps/data/str-features-encoder-{features_name}.pkl").expanduser()
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder for {features_name} not found at {encoder_path}"
        )

    encoder = surrogate.load_index(encoder_path)
    loaded_encoders[features_name] = encoder

    return encoder


class EncodeRequest(BaseModel):
    features_name: str
    feature_vector: list[float]


def create_app() -> FastAPI:
    app = FastAPI(title="str-features encoder")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/encode")
    def encode(request: EncodeRequest):
        if not request.features_name:
            raise HTTPException(status_code=400, detail="features_name is required")

        if not request.feature_vector:
            raise HTTPException(status_code=400, detail="feature_vector is required")

        feature_vector = np.atleast_2d(request.feature_vector).astype(np.float32)
        encoder = load_encoder(request.features_name)

        tf = encoder.encode(feature_vector, inverted=False, query=True)
        str_doc = surrogate.generate_documents(tf, compact=True, delimiter="^")
        str_doc = next(iter(str_doc))

        return {"feature_str": str_doc}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="str-features encoder")
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
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
