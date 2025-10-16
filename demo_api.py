"""
FastAPI demo API for ArtiMuse.

功能：
- 常驻加载模型，提供图像审美 7 维度打分、总分与评语的 HTTP 接口。

注意：
- 日志必须使用英文；代码注释使用中文。
- 禁止函数内 import；避免不必要的 try/except。
"""

from __future__ import annotations

import io
import os
import logging
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.demo_utils.model_server import (
    ModelServer,
    AESTHETIC_DIMENSIONS,
)


# ------------------------ 常量与全局 ------------------------
DEFAULT_CHECKPOINT: str = os.environ.get("ARTIMUSE_CKPT", "checkpoints/ArtiMuse")
DEFAULT_DEVICE: str = os.environ.get("ARTIMUSE_DEVICE", "cuda:0")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("demo_api")


# 无需本地 ModelServer 与预处理，直接复用抽象


# ------------------------ FastAPI 应用 ------------------------

APP = FastAPI(title="ArtiMuse Demo API", version="1.0.0")
SERVER = ModelServer(ckpt_dir=DEFAULT_CHECKPOINT, device=DEFAULT_DEVICE)


@APP.get("/health")
def health() -> Dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@APP.get("/meta")
def meta() -> Dict[str, str | List[str]]:
    """返回元信息：维度名称、设备与检查点。"""
    return {
        "dimensions": AESTHETIC_DIMENSIONS,
        "device": DEFAULT_DEVICE,
        "checkpoint": DEFAULT_CHECKPOINT,
    }


@APP.post("/infer")
def infer(file: UploadFile = File(...)) -> JSONResponse:
    """推理接口：上传图片，返回分数与评语。"""
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
    """启动 Uvicorn 服务。"""
    uvicorn.run(APP, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
