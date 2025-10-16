"""
FastAPI demo API for ArtiMuse.

Features:
- Persistent model service with HTTP endpoints for 7-dimension scores,
  total score, and per-dimension comments.

Notes:
- Logging is English; comments are English.
- No in-function imports; avoid unnecessary try/except blocks.
"""

from __future__ import annotations

import io
import os
import logging
from typing import Dict, List
import argparse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from artimuse.demo_utils.model_server import (
    ModelServer,
    AESTHETIC_DIMENSIONS,
)


# ------------------------ Constants ------------------------
DEFAULT_CHECKPOINT: str = os.environ.get("ARTIMUSE_CKPT", "checkpoints/ArtiMuse")
DEFAULT_DEVICE: str = os.environ.get("ARTIMUSE_DEVICE", "cuda:0")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("demo_api")


# No local ModelServer or preprocessing needed here; reuse the shared abstraction


# ------------------------ FastAPI App ------------------------

APP = FastAPI(title="ArtiMuse Demo API", version="1.0.0")
SERVER = ModelServer(ckpt_dir=DEFAULT_CHECKPOINT, device=DEFAULT_DEVICE)


@APP.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@APP.get("/meta")
def meta() -> Dict[str, str | List[str]]:
    """Return meta info: dimensions, device, checkpoint."""
    return {
        "dimensions": AESTHETIC_DIMENSIONS,
        "device": DEFAULT_DEVICE,
        "checkpoint": DEFAULT_CHECKPOINT,
    }


@APP.post("/infer")
def infer(file: UploadFile = File(...)) -> JSONResponse:
    """Inference endpoint: upload an image and get scores and comments."""
    content = file.file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    aspect_scores, total_score, aspect_comments = SERVER.score_and_comment(img)
    payload = {
        "total_score": round(float(total_score), 3),
        "aspect_scores": {k: round(float(v), 3) for k, v in aspect_scores.items()},
        "aspect_comments": aspect_comments,
    }
    return JSONResponse(payload)


def main() -> None:
    """Launch Uvicorn service with configurable host/port."""
    parser = argparse.ArgumentParser(description="Run ArtiMuse FastAPI demo server")
    parser.add_argument("--host", "--listen", dest="host", default=os.environ.get("HOST", "0.0.0.0"), help="Host/IP to bind")
    parser.add_argument("--port", dest="port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to bind")
    args = parser.parse_args()

    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
