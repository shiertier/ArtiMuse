"""
Gradio demo for ArtiMuse.

功能：
- 允许上传图像，进行推理。
- 输出 7 个维度的分数、总分数，以及每个维度的评语。
- 使用雷达图（蜘蛛网图）美观展示 7 维度分数（0-100，间隔 10）。
- 支持连续多次推理，模型常驻内存。

注意：
- 日志必须使用英文；代码注释使用中文。
- 禁止函数内 import；避免不必要的 try/except。
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.demo_utils.model_server import (
    ModelServer,
    AESTHETIC_DIMENSIONS,
)


# ------------------------ 常量与全局 ------------------------

DEFAULT_CHECKPOINT: str = os.environ.get("ARTIMUSE_CKPT", "checkpoints/ArtiMuse")
DEFAULT_DEVICE: str = os.environ.get("ARTIMUSE_DEVICE", "cuda:0")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("demo_gradio")


# 引入以声明依赖（打包工具可据此收集）
from src.demo_utils.model_server import to_model_tensor  # noqa: F401


# ------------------------ 可视化：雷达图 ------------------------

def _radar_figure(aspect_scores: Dict[str, float]) -> plt.Figure:
    """绘制 7 维度雷达图，半径 0~100，每 10 一根网格线。"""
    labels = list(aspect_scores.keys())
    values = [aspect_scores[k] for k in labels]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # 闭合多边形
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7), facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 网格与坐标
    grid_vals = list(range(0, 101, 10))
    ax.set_rgrids(grid_vals, angle=0, color="#AAAAAA", alpha=0.5, fontsize=8)
    ax.set_ylim(0, 100)

    # 轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment("center")

    # 多边形
    ax.plot(angles, values, color="#6C63FF", linewidth=2)
    ax.fill(angles, values, color="#6C63FF", alpha=0.25)

    # 美化
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.set_facecolor("#FAFAFA")
    return fig


# ------------------------ Gradio 回调 ------------------------

SERVER = ModelServer(ckpt_dir=DEFAULT_CHECKPOINT, device=DEFAULT_DEVICE)


def _run_infer(image: Image.Image) -> Tuple[plt.Figure, str, str]:
    """Gradio 处理函数：返回雷达图、分数汇总文本、评语文本。"""
    if image is None:
        raise ValueError("No image provided.")

    aspect_scores, total_score, aspect_comments = SERVER.score_and_comment(image)
    fig = _radar_figure(aspect_scores)

    # 分数文本
    score_lines = [f"Total Score: {total_score:.1f}"]
    for k in AESTHETIC_DIMENSIONS:
        score_lines.append(f"{k}: {aspect_scores[k]:.1f}")
    score_text = "\n".join(score_lines)

    # 评语文本
    comment_lines: List[str] = []
    for k in AESTHETIC_DIMENSIONS:
        comment_lines.append(f"[{k}]\n{aspect_comments[k]}\n")
    comment_text = "\n".join(comment_lines)
    return fig, score_text, comment_text


def launch() -> None:
    """启动 Gradio 界面。"""
    css = """
    .gradio-container {max-width: 1200px}
    .score-box textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;}
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <div style="text-align:center">
              <h1 style="margin-bottom:8px">ArtiMuse – Aesthetic Assessment Demo</h1>
              <p style="color:#666; margin-top:0">Upload an image to get 7-dim scores, total score and comments.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Input Image", height=420)
                run_btn = gr.Button("Run Inference", variant="primary")
            with gr.Column(scale=1):
                fig_out = gr.Plot(label="Radar Chart (0-100)")
                score_out = gr.Textbox(
                    label="Scores",
                    lines=10,
                    interactive=False,
                    elem_classes=["score-box"],
                )
            with gr.Column(scale=1):
                comment_out = gr.Textbox(
                    label="Comments",
                    lines=20,
                    interactive=False,
                )

        run_btn.click(_run_infer, inputs=[image_in], outputs=[fig_out, score_out, comment_out])

    demo.queue(concurrency_count=2).launch()


if __name__ == "__main__":
    launch()
