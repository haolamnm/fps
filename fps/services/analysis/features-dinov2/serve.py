import argparse
import io
import urllib.request
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is not available*"
)


class DinoV2FrameEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(  # type: ignore
            self.device
        )
        self.model = model.eval()

        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def encode_pil(self, pil_image: Image.Image) -> list[float]:
        """Encodes a PIL image to a feature vector."""
        with torch.no_grad():
            inputs = self.transform(pil_image).unsqueeze(0).to(self.device)  # type: ignore
            image_features = self.model(inputs)
            image_features = F.normalize(image_features, dim=-1, p=2)

        return image_features.squeeze().cpu().numpy().tolist()

    def encode_path(self, image_path: Path) -> list[float]:
        """Encodes an image from a file path to a feature vector."""
        pil_image = Image.open(image_path).convert("RGB")
        return self.encode_pil(pil_image)

    def encode_url(self, image_url: str) -> list[float]:
        """Encodes an image from a URL to a feature vector."""
        with urllib.request.urlopen(image_url) as response:
            pil_image = Image.open(response).convert("RGB")
        return self.encode_pil(pil_image)


def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="dinov2 encoder")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    encoder = DinoV2FrameEncoder(model_name)
    app.state.encoder = encoder

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.get("/encode")
    def encode_url(url: str):
        if not url:
            raise HTTPException(status_code=400, detail="Missing 'url' parameter")
        try:
            feature_vector = encoder.encode_url(url)
            return {"url": url, "feature_vector": feature_vector, "length": len(feature_vector)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/encode")
    async def encode_image(image: UploadFile = File(...)):
        if not image:
            raise HTTPException(status_code=400, detail="Missing 'image' file")

        try:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            feature_vector = encoder.encode_pil(pil_image)
            return {
                "filename": image.filename,
                "feature_vector": feature_vector,
                "length": len(feature_vector),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dinov2 encoder service")
    parser.add_argument(
        "--model",
        default="dinov2_vitl14",
        type=str,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="name of the DINOv2 model to use",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="host to run the service on",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="port to run the service on",
    )
    args = parser.parse_args()

    app = create_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)
