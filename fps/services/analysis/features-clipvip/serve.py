import argparse

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class CLIPVIPQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = self.model.eval()
        self.model = torch.compile(self.model)

    def encode(self, query: str) -> list[float]:
        with torch.no_grad():
            tokens = self.tokenizer(query, return_tensors="pt", padding=True)
            embeddings = self.model.get_text_features(**tokens)  # type: ignore
            embeddings = F.normalize(embeddings, dim=-1, p=2)
            return embeddings.squeeze().cpu().numpy().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="clipvip encoder")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    encoder = CLIPVIPQueryEncoder(model_name)
    app.state.encoder = encoder

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/encode")
    def encode(request: QueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        try:
            feature_vector = encoder.encode(query)
            return {"feature_vector": feature_vector, "length": len(feature_vector)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clipvip encoder")
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch16",
        type=str,
        choices=[
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ],
        help="model name to use for encoding queries",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="port to run the service on",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="host to run the service on",
    )
    args = parser.parse_args()

    app = create_app(args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)
