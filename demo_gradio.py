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


# ------------------------ Visualization: Radar Chart with Scores ------------------------

def _radar_figure(aspect_scores: Dict[str, float]) -> plt.Figure:
    """Draw a 7-dimension radar chart (0–100, grid step 10) with score labels."""
    labels = list(aspect_scores.keys())
    values = [aspect_scores[k] for k in labels]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the polygon
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid and axes
    grid_vals = list(range(0, 101, 10))
    ax.set_rgrids(grid_vals, angle=0, color="#AAAAAA", alpha=0.5, fontsize=8)
    ax.set_ylim(0, 100)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment("center")

    # Polygon
    ax.plot(angles_closed, values_closed, color="#6C63FF", linewidth=2.5, marker="o", markersize=8)
    ax.fill(angles_closed, values_closed, color="#6C63FF", alpha=0.25)

    # Add score labels on each point
    for angle, value, label in zip(angles, values, labels):
        ax.text(angle, value + 8, f"{value:.1f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#6C63FF",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#6C63FF", alpha=0.8))

    # Styling
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    return fig


# ------------------------ Gradio Handlers ------------------------

SERVER = ModelServer(ckpt_dir=DEFAULT_CHECKPOINT, device=DEFAULT_DEVICE)


def _run_infer(image: Image.Image) -> Tuple[plt.Figure, str, List[List[str]]]:
    """Gradio handler: return radar chart, total score, and per-dimension comments table."""
    if image is None:
        raise ValueError("No image provided.")

    aspect_scores, total_score, aspect_comments = SERVER.score_and_comment(image)
    fig = _radar_figure(aspect_scores)

    # Total score text (centered, large)
    total_score_text = f"Total Score: {total_score:.1f}/100"

    # Comments table: each row is [dimension, score, comment]
    comments_table = []
    for k in AESTHETIC_DIMENSIONS:
        comments_table.append([k, f"{aspect_scores[k]:.1f}", aspect_comments[k]])

    return fig, total_score_text, comments_table


def launch(server_name: str, server_port: int) -> None:
    """Launch Gradio UI with centered layout and fixed heights."""
    css = """
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header-section {
        text-align: center;
        margin-bottom: 30px;
    }
    /* Top section: ensure equal heights and alignment */
    .top-row {
        display: flex !important;
        justify-content: center !important;
        gap: 40px !important;
        margin-bottom: 40px !important;
    }
    .image-column {
        flex: 0 0 450px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }
    .chart-column {
        flex: 0 0 450px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 20px !important;
    }
    /* Fix plot height */
    .chart-column .gradio-plot {
        height: 450px !important;
        min-height: 450px !important;
    }
    .total-score-box {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-size: 28px;
        font-weight: bold;
        width: 100%;
        box-sizing: border-box;
    }
    /* Comments table section */
    .comments-section {
        width: 900px !important;
        margin: 0 auto !important;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown(
            """
            <div class="header-section">
              <h1 style="margin-bottom:8px">🎨 ArtiMuse – Aesthetic Assessment</h1>
              <p style="color:#666; margin-top:0; font-size:16px">Upload an image to get comprehensive aesthetic evaluation with 7-dimensional scores and detailed comments.</p>
            </div>
            """
        )

        # Top section: Image input (left) + Chart + Total Score (right)
        with gr.Row(elem_classes=["top-row"]):
            # Left: Image input
            with gr.Column(scale=0, min_width=450, elem_classes=["image-column"]):
                gr.Markdown("### Input Image")
                image_in = gr.Image(type="pil", label="", height=450)
                run_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg", scale=1)

            # Right: Chart and Total Score
            with gr.Column(scale=0, min_width=450, elem_classes=["chart-column"]):
                gr.Markdown("### Aesthetic Evaluation")
                fig_out = gr.Plot(label="", show_label=False)
                total_score_out = gr.Markdown(
                    value="Total Score: --/100",
                    elem_classes=["total-score-box"]
                )

        # Bottom section: Comments table (centered and same width as top section)
        gr.Markdown("### Detailed Evaluation")
        comments_table = gr.Dataframe(
            headers=["Dimension", "Score", "Comment"],
            label="",
            interactive=False,
            wrap=True,
            elem_classes=["comments-section"]
        )

        # Connect inference button
        def _run_infer_wrapper(image):
            fig, total_score_text, comments_data = _run_infer(image)

            # Prepare total score markdown
            total_score_md = f"""
            <div class="total-score-box">
                {total_score_text}
            </div>
            """

            # Return: fig, total_score_md, comments_table
            return [fig, total_score_md, comments_data]

        run_btn.click(
            _run_infer_wrapper,
            inputs=[image_in],
            outputs=[fig_out, total_score_out, comments_table]
        )

    demo.queue(max_size=2).launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ArtiMuse Gradio demo UI")
    parser.add_argument("--host", "--listen", dest="host", default=os.environ.get("HOST", "0.0.0.0"), help="Host/IP to bind")
    parser.add_argument("--port", dest="port", type=int, default=int(os.environ.get("PORT", 7860)), help="Port to bind")
    args = parser.parse_args()
    launch(args.host, args.port)
