import argparse
import sys

import open_clip
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class OpenCLIPQueryEncoder:
    def __init__(self, model_name: str, pretrained: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.context_length = self.model.context_length
        self.model = torch.compile(self.model)

    def encode(self, query: str) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                context_length=self.context_length,  # type: ignore
            ).to(self.device)
            features = self.model.encode_text(inputs).float()  # type: ignore
            features = torch.nn.functional.normalize(features, dim=-1, p=2)

            return features.cpu().squeeze().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(model_name: str, pretrained: str, title: str) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    encoder = OpenCLIPQueryEncoder(model_name, pretrained)
    app.state.encoder = encoder

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.get("/get-text-feature")
    def encode(request: QueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        try:
            feature_vector = encoder.encode(query)
            return feature_vector
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="openclip encoder")
    parser.add_argument(
        "--model-name",
        default="ViT-L-14",
        type=str,
        choices=["ViT-L-14", "ViT-B-32", "ViT-B-16"],
        help="model name to use for feature extraction (default: ViT-L-14)",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s32b_b82k",
        type=str,
        choices=[
            "laion2b_s32b_b82k",
            "datacomp_xl_s13b_b90k",
        ],
        help="pretrained model to use for feature extraction (default: laion2b_s32b_b82k)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to run the server on (default: locahost)",
    )
    args = parser.parse_args()

    return args


def main() -> int:
    args = parse_args()
    title = "clip-laion encoder" if args.pretrained == "laion2b_s32b_b82k" else "clip-datacomp encoder"

    app = create_app(args.model_name, args.pretrained, title)
    uvicorn.run(app, host=args.host, port=args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
