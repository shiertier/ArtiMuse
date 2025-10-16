<h1 style="line-height: 1.4;">
  <span style="color: #FF3E3E;">A</span><span style="color: #FF914D;">r</span><span 
  style="color: #FFC94D;">t</span><span style="color: #B6E24D;">i</span><span
  style="color: #4DDC95;">M</span><span style="color: #4DB8FF;">u</span><span
  style="color: #8564FF;">s</span><span style="color: #C74DFF;">e</span>:
  Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding
</h1>


<!-- <h1 style="margin-top: -10px; color: #666; font-weight: normal; font-size: 20px;">
  书生 · 妙析多模态美学理解大模型
</h1> -->

<div align="center">

\[[🌐 Project Page](https://thunderbolt215.github.io/ArtiMuse-project/)]
\[[🖥️ Online Demo](http://artimuse.intern-ai.org.cn/)]
\[[📄 Paper](https://arxiv.org/abs/2507.14533)]
\[[🧩 Checkpoints: 🤗 [Hugging Face](https://huggingface.co/collections/Thunderbolt215215/artimuse-68b7d2c7137d8ed119c8774e) | 🤖 [ModelScope](https://modelscope.cn/collections/ArtiMuse-abea7a7922274d)]]
</div>


![Online Demo QR Code](assets/images/QRcode.jpg)


> 🔬 **We are actively developing an enhanced version of ArtiMuse with reasoning capabilities — _ArtiMuse-R1_.**  
> 🌟 Stay tuned for exciting updates and improvements!


**Shuo Cao**, **Nan Ma**, **Jiayang Li**, **Xiaohui Li**, **Lihao Shao**, **Kaiwen Zhu**, **Yu Zhou**, **Yuandong Pu**, **Jiarui Wu**, **Jiaquan Wang**, **Bo Qu**, **Wenhai Wang**, **Yu Qiao**, **Dajuin Yao†**, **Yihao Liu†**

University of Science and Technology of China, Shanghai AI Laboratory, China Academy of Art, Peking University 

† Corresponding Authors


![Teaser](assets/images/Teaser.jpg "Teaser Figure")


## 📰 News & Updates

- 🚀 **Sep 3, 2025**  
  The **Checkpoints** and **Evaluation Code** of ArtiMuse are now available! 🚀

- 🚀 **July 28, 2025**  
  **ArtiMuse** was officially released at **WAIC 2025**, in the forum _"Evolving with AI: The Iteration and Resilience of Artistic Creativity"_

- 🚀 **July 24, 2025**  
  The **Online Demo** is now open for public access!

- 🚀 **July 21, 2025**  
  The **Paper**, **Repository** and **Project Page** are now live!


## 🔍 Abstract

The rapid advancement of educational applications, artistic creation, and AI-generated content (AIGC) technologies has substantially increased practical requirements for comprehensive Image Aesthetics Assessment (IAA), particularly demanding methods capable of delivering both quantitative scoring and professional understanding.  
 
In this paper, we present:  
**(1) ArtiMuse**, an innovative MLLM-based IAA model with Joint Scoring and Expert-Level Understanding capabilities;  
**(2) ArtiMuse-10K**, the first expert-curated image aesthetic dataset comprising 10,000 images spanning 5 main categories and 15 subcategories, each annotated by professional experts with 8-dimensional attributes analysis and a holistic score.  


## 📦 Checkpoints

All paper-version checkpoints share the same **text pretraining process**, but differ in their **score finetuning datasets**:

| Checkpoint             | Score Finetuning Dataset | Download | Notes |
|-------------------------|--------------------------|----------|-------|
| `ArtiMuse`              | ArtiMuse-10K             | [🤗HF link](https://huggingface.co/Thunderbolt215215/ArtiMuse)<br>[🤖MS link](https://modelscope.cn/models/thunderbolt/ArtiMuse) | **Paper Version (Recommended)** |
| `ArtiMuse_AVA`          | AVA                      | [🤗HF link](https://huggingface.co/Thunderbolt215215/ArtiMuse_AVA)<br>[🤖MS link](https://modelscope.cn/models/thunderbolt/ArtiMuse_AVA) | Paper Version |
| `ArtiMuse_FLICKR-AES`   | FLICKR-AES               | [🤗HF link](https://huggingface.co/Thunderbolt215215/ArtiMuse_FLICKR-AES)<br> [🤖MS link](https://modelscope.cn/models/thunderbolt/ArtiMuse_FLICKR-AES) | Paper Version |
| `ArtiMuse_PARA`         | PARA                     | [🤗HF link](https://huggingface.co/Thunderbolt215215/ArtiMuse_PARA)<br> [🤖MS link](https://modelscope.cn/models/thunderbolt/ArtiMuse_PARA) | Paper Version |
| `ArtiMuse_TAD66K`       | TAD66K                   | [🤗HF link](https://huggingface.co/Thunderbolt215215/ArtiMuse_TAD66K)<br> [🤖MS link](https://modelscope.cn/models/thunderbolt/ArtiMuse_TAD66K) | Paper Version |
| `ArtiMuse_OnlineDemo`   | ArtiMuse-10K & Internal Datasets  |  —   | Surpasses paper versions thanks to additional internal datasets and advanced training; also supports fine-grained attribute scores. For access, please contact us for business collaboration. |
| `ArtiMuse-R1`           |    —       |  —  | Next-generation model trained with GRPO, supporting CoT reasoning, delivering more accurate score predictions, and extending beyond IAA to handle a wider range of tasks. |

## ⚙️ Setup

Clone this repository:

```
git clone https://github.com/thunderbolt215/ArtiMuse.git
```
Create a conda virtual environment and activate it: (please ensure that `Python>=3.9`).

```
conda create -n artimuse python=3.10
conda activate artimuse
```

Install dependencies using `requirements.txt`:
```
pip install -r requirements.txt
```
We recommend to use FlashAttention for acceleration:
```
pip install flash-attn --no-build-isolation
```

## 📊 Evaluation

### 1. Prepare Checkpoints

Download the pretrained checkpoints and place them under the `checkpoints/` directory.
The folder structure should look like:

```
ArtiMuse
└── checkpoints/
    ├── ArtiMuse
    ├── ArtiMuse_AVA
    ├── ArtiMuse_FLICKR-AES
    ├── ...
```

---

### 2. Evaluation on a Single Image

Run the following command to evaluate a single image:

```bash
python src/eval/eval_image.py \
    --model_name ArtiMuse \
    --image_path example/test.jpg \
    --device cuda:0
```

* **Arguments**

  * `--model_name`: Name of the checkpoint to use (e.g., `ArtiMuse`, `ArtiMuse_AVA`).
  * `--image_path`: Path to the input image.
  * `--device`: Inference device, e.g., `cuda:0`.

* **Results**
  are saved to:

  ```
  results/image_results/{input_image_name}_{model_name}_eval.json
  ```

---

### 3. Evaluation on Benchmark Datasets

Download the test datasets and organize them under `test_datasets/{dataset_name}/images/`.
The expected structure is:

```
ArtiMuse
└── test_datasets/
    ├── AVA
    │   ├── images/
    │   └── test.json
    ├── TAD66K
    ├── FLICKR-AES
    └── ...
```

* `images/`: contains the test images.
* `test.json`: provides the ground-truth scores (`gt_score`) for evaluation.

Run dataset-level evaluation with:

```bash
python src/eval/eval_dataset.py \
    --model_name ArtiMuse_AVA \
    --dataset AVA \
    --device cuda:0
```

* **Arguments**

  * `--model_name`: Name of the checkpoint to use (e.g., `ArtiMuse_AVA`).
  * `--dataset`: Dataset name (e.g., `AVA`, `TAD66K`, `FLICKR-AES`).
  * `--device`: Inference device.

* **Results**
   are saved to:

  ```
  results/dataset_results/{dataset}_{model_name}.json
  ```

## 🧪 Demos

### Installation

First, install the package locally so imports resolve cleanly:

```bash
pip install -e .
```

This will install ArtiMuse in editable mode, making the `artimuse` package importable from anywhere in your Python environment.

### Project Structure

The package is organized as follows:

```
ArtiMuse/
├── src/
│   ├── artimuse/                    # Main package
│   │   ├── demo_utils/              # Shared demo utilities
│   │   │   └── model_server.py      # Persistent model service
│   │   └── internvl/                # InternVL model components
│   │       ├── conversation.py
│   │       └── model/
│   │           ├── internvl_chat/   # ArtiMuse model
│   │           ├── internlm2/       # InternLM2 components
│   │           └── phi3/            # Phi3 components
│   ├── eval/                        # Evaluation scripts
│   │   ├── eval_image.py
│   │   └── eval_dataset.py
│   └── demo_utils/                  # Additional demo utilities
├── demo_gradio.py                   # Gradio UI entry point
├── demo_api.py                      # FastAPI entry point
└── checkpoints/                     # Model checkpoints directory
```

### Usage Methods

#### Method 1: Gradio UI (Recommended for Interactive Use)

Interactive web interface with image upload, 7-dimensional radar chart, and aesthetic comments.

**Installation:**
```bash
pip install -r requirements_gradio.txt
# or
pip install .[gradio]
```

**Run:**
```bash
python demo_gradio.py --host 0.0.0.0 --port 7860
```

Then open your browser to `http://localhost:7860`

**Options:**
- `--host`: Server host (default: `0.0.0.0`)
- `--port`: Server port (default: `7860`)

**Environment variables (optional):**
- `ARTIMUSE_CKPT`: Path to checkpoint (default: `checkpoints/ArtiMuse`)
- `ARTIMUSE_DEVICE`: Device string (default: `cuda:0`)

---

#### Method 2: FastAPI Service (Recommended for Production)

RESTful API service with persistent model loading for efficient batch inference.

**Installation:**
```bash
pip install -r requirements_api.txt
# or
pip install .[api]
```

**Run:**
```bash
python demo_api.py --host 0.0.0.0 --port 8000
```

**API Endpoints:**

- `GET /health` - Health check
  ```bash
  curl http://localhost:8000/health
  ```

- `GET /meta` - Get model metadata (dimensions, device, checkpoint)
  ```bash
  curl http://localhost:8000/meta
  ```

- `POST /infer` - Run inference on an image
  ```bash
  curl -X POST -F "file=@image.jpg" http://localhost:8000/infer
  ```

  **Response:**
  ```json
  {
    "total_score": 75.5,
    "aspect_scores": {
      "Composition & Design": 78.2,
      "Visual Elements & Structure": 76.1,
      "Technical Execution": 74.3,
      "Originality & Creativity": 72.5,
      "Theme & Communication": 75.8,
      "Emotion & Viewer Response": 77.2,
      "Overall Gestalt": 76.0
    },
    "aspect_comments": {
      "Composition & Design": "The composition is well-balanced...",
      ...
    }
  }
  ```

**Options:**
- `--host`: Server host (default: `0.0.0.0`)
- `--port`: Server port (default: `8000`)

**Environment variables (optional):**
- `ARTIMUSE_CKPT`: Path to checkpoint (default: `checkpoints/ArtiMuse`)
- `ARTIMUSE_DEVICE`: Device string (default: `cuda:0`)

---

#### Method 3: Python API (For Custom Integration)

Use the `ModelServer` class directly in your Python code for custom workflows.

**Example:**
```python
from artimuse.demo_utils.model_server import ModelServer
from PIL import Image

# Initialize the model server
server = ModelServer(
    ckpt_dir="checkpoints/ArtiMuse",
    device="cuda:0"
)

# Load and evaluate an image
image = Image.open("path/to/image.jpg")
aspect_scores, total_score, aspect_comments = server.score_and_comment(image)

# Access results
print(f"Total Score: {total_score}")
for dimension, score in aspect_scores.items():
    print(f"{dimension}: {score}")
    print(f"Comment: {aspect_comments[dimension]}\n")
```

**Available Methods:**
- `score_and_comment(img)` - Get 7-dimensional scores, total score, and comments
  - Returns: `(aspect_scores: Dict[str, float], total_score: float, aspect_comments: Dict[str, str])`

---

### Configuration

**Environment variables (optional):**
- `ARTIMUSE_CKPT`: Path to checkpoint directory (default: `checkpoints/ArtiMuse`)
- `ARTIMUSE_DEVICE`: CUDA device string (default: `cuda:0`)
- `HOST`: Server host for demos (default: `0.0.0.0`)
- `PORT`: Server port for demos (default: `7860` for Gradio, `8000` for FastAPI)

**Example:**
```bash
export ARTIMUSE_CKPT=checkpoints/ArtiMuse_AVA
export ARTIMUSE_DEVICE=cuda:1
python demo_gradio.py
```

## 🙏 Acknowledgements

Our work is built upon the [InternVL-3](https://github.com/OpenGVLab/InternVL) model as the base foundation. We also refer to the implementation of [Q-Align](https://github.com/Q-Future/Q-Align) during development. We sincerely thank the authors of both projects for their excellent contributions to the community.


## ✒️ Citation

If you find this work useful, please consider citing:

```bibtex
@misc{cao2025artimusefinegrainedimageaesthetics,
      title={ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding}, 
      author={Shuo Cao and Nan Ma and Jiayang Li and Xiaohui Li and Lihao Shao and Kaiwen Zhu and Yu Zhou and Yuandong Pu and Jiarui Wu and Jiaquan Wang and Bo Qu and Wenhai Wang and Yu Qiao and Dajuin Yao and Yihao Liu},
      year={2025},
      eprint={2507.14533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.14533}, 
}
```
