import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .clip2video.modules.tokenization_clip import SimpleTokenizer as CLIPTokenizer
from .config import Config
from .utils import get_device, load_model


class CLIP2VideoQueryEncoder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.gpu = torch.cuda.is_available()
        self.device, self.num_gpu = get_device(self.config, local_rank=self.config.local_rank)

        self.tokenizer = CLIPTokenizer()
        self.SPECIAL_TOKENS = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }
        self.model = load_model(self.config, self.device)
        self.model.eval()
        self.model = torch.compile(self.model)

    def preprocess(self, query: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words: list[str] = self.tokenizer.tokenize(query)

        # Add CLS token
        words = [self.SPECIAL_TOKENS["CLS_TOKEN"]] + words
        length = self.config.max_words - 1
        if len(words) > length:
            words = words[:length]

        # Add SEP token
        words.append(self.SPECIAL_TOKENS["SEP_TOKEN"])

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # Add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.config.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # Ensure the length of feature to be equal with max words
        assert len(input_ids) == self.config.max_words
        assert len(input_mask) == self.config.max_words
        assert len(segment_ids) == self.config.max_words
        pairs_text = torch.LongTensor(input_ids)
        pairs_mask = torch.LongTensor(input_mask)
        pairs_segment = torch.LongTensor(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def encode(self, query: str) -> list[float]:
        input_ids, input_mask, segment_ids = self.preprocess(query)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        segment_ids = segment_ids.unsqueeze(0).to(self.device)
        input_mask = input_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            sequence_output = self.model.get_sequence_output(  # type: ignore
                input_ids, segment_ids, input_mask
            )
            features = self.model.get_text_features(sequence_output, input_mask)  # type: ignore
            return features.squeeze().cpu().numpy().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="clip2video encoder")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    encoder = CLIP2VideoQueryEncoder(config)
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
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip2video encoder")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to run on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run on",
    )
    args = parser.parse_args()

    config = Config(
        video_path=None,
        checkpoint_dir="checkpoint",
        clip_path="checkpoint/ViT-B-32.pt",
    )

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)
