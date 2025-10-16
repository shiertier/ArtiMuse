"""
Gradio demo for ArtiMuse.

Features:
- Upload an image and run inference.
- Display 7-dimension scores, total score, and per-dimension comments.
- Render a radar chart (0–100 radius, grid every 10).
- Support multiple inferences with a resident model instance.

Notes:
- Logging is English; comments are English.
- No in-function imports; avoid unnecessary try/except blocks.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple
import argparse

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from artimuse.demo_utils.model_server import (
    ModelServer,
    AESTHETIC_DIMENSIONS,
)


# ------------------------ Constants ------------------------

DEFAULT_CHECKPOINT: str = os.environ.get("ARTIMUSE_CKPT", "checkpoints/ArtiMuse")
DEFAULT_DEVICE: str = os.environ.get("ARTIMUSE_DEVICE", "cuda:0")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("demo_gradio")


# Import solely to declare dependency for packagers
from artimuse.demo_utils.model_server import to_model_tensor  # noqa: F401


# ------------------------ Visualization: Radar Chart ------------------------

def _radar_figure(aspect_scores: Dict[str, float]) -> plt.Figure:
    """Draw a 7-dimension radar chart (0–100, grid step 10)."""
    labels = list(aspect_scores.keys())
    values = [aspect_scores[k] for k in labels]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the polygon
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7), facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid and axes
    grid_vals = list(range(0, 101, 10))
    ax.set_rgrids(grid_vals, angle=0, color="#AAAAAA", alpha=0.5, fontsize=8)
    ax.set_ylim(0, 100)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment("center")

    # Polygon
    ax.plot(angles, values, color="#6C63FF", linewidth=2)
    ax.fill(angles, values, color="#6C63FF", alpha=0.25)

    # Styling
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.set_facecolor("#FAFAFA")
    return fig


# ------------------------ Gradio Handlers ------------------------

SERVER = ModelServer(ckpt_dir=DEFAULT_CHECKPOINT, device=DEFAULT_DEVICE)


def _run_infer(image: Image.Image) -> Tuple[plt.Figure, str, str]:
    """Gradio handler: return radar chart, score summary, and comments."""
    if image is None:
        raise ValueError("No image provided.")

    aspect_scores, total_score, aspect_comments = SERVER.score_and_comment(image)
    fig = _radar_figure(aspect_scores)

    # Score text
    score_lines = [f"Total Score: {total_score:.1f}"]
    for k in AESTHETIC_DIMENSIONS:
        score_lines.append(f"{k}: {aspect_scores[k]:.1f}")
    score_text = "\n".join(score_lines)

    # Comments text
    comment_lines: List[str] = []
    for k in AESTHETIC_DIMENSIONS:
        comment_lines.append(f"[{k}]\n{aspect_comments[k]}\n")
    comment_text = "\n".join(comment_lines)
    return fig, score_text, comment_text


def launch(server_name: str, server_port: int) -> None:
    """Launch Gradio UI."""
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

    demo.queue(max_size=2).launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ArtiMuse Gradio demo UI")
    parser.add_argument("--host", "--listen", dest="host", default=os.environ.get("HOST", "0.0.0.0"), help="Host/IP to bind")
    parser.add_argument("--port", dest="port", type=int, default=int(os.environ.get("PORT", 7860)), help="Port to bind")
    args = parser.parse_args()
    launch(args.host, args.port)
