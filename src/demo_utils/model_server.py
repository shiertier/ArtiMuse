"""
Shared model server for ArtiMuse demos.

职责：
- 常驻加载 ArtiMuse 模型与分词器
- 统一提供 7 维度打分、总分与评语
- 暴露最小且稳定的接口，供 Gradio / API 复用

注意：
- 日志使用英文；注释使用中文
- 禁止函数内 import；不添加多余的异常分支
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from src.artimuse.internvl.model.internvl_chat.aes_tokens import AESTHETICS_TOKEN_LIST
from src.artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel
from src.artimuse.internvl.conversation import get_conv_template


# ------------------------ 常量 ------------------------

AESTHETIC_DIMENSIONS: List[str] = [
    "Composition & Design",
    "Visual Elements & Structure",
    "Technical Execution",
    "Originality & Creativity",
    "Theme & Communication",
    "Emotion & Viewer Response",
    "Overall Gestalt",
]

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

LOGGER = logging.getLogger("artimuse.model_server")


# ------------------------ 预处理 ------------------------

def build_transform(input_size: int = 448) -> T.Compose:
    """构建图像预处理流水线。"""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def to_model_tensor(img: Image.Image, input_size: int = 448) -> torch.Tensor:
    """将 PIL.Image 转换为模型输入张量 (1, C, H, W)。"""
    transform = build_transform(input_size)
    return transform(img).unsqueeze(0)


# ------------------------ 模型常驻服务 ------------------------

class ModelServer:
    """ArtiMuse 模型服务：统一暴露打分与评语接口。"""

    def __init__(self, ckpt_dir: str, device: str) -> None:
        LOGGER.info("Loading model from %s on %s", ckpt_dir, device)
        self.device = device
        self.model = (
            InternVLChatModel.from_pretrained(
                ckpt_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
            )
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir, trust_remote_code=True, use_fast=False
        )
        self.generation_config: Dict[str, int | bool] = {
            "max_new_tokens": 2048,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # 0..100 的权重用于 softmax 后的期望值
        self._weights = torch.arange(0, 101, dtype=torch.float32, device=device)

    def _score_questions(self, questions: List[str], pixel_values: torch.Tensor) -> List[float]:
        """批量打分：输入一组问题，输出 [0,100] 浮点分数列表。"""
        assert pixel_values.ndim == 4 and pixel_values.size(0) == 1

        num_q = len(questions)
        num_patches_list = [pixel_values.size(0)] * num_q
        pixel_values_batch = torch.cat((pixel_values,) * num_q, dim=0)

        # 构建 queries
        queries: List[str] = []
        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_ctx_id
        for q in questions:
            question = "<image>\n" + q if "<image>" not in q else q
            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches_list[0] + "</img>"
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        # 批量 tokenization
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        eos_token_id = self.tokenizer.convert_tokens_to_ids(
            get_conv_template(self.model.template).sep.strip()
        )
        gen_cfg = dict(self.generation_config)
        gen_cfg["eos_token_id"] = eos_token_id

        outputs = self.model.generate_logits(
            pixel_values=pixel_values_batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_cfg,
        )
        logits = outputs.logits
        pref_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in AESTHETICS_TOKEN_LIST]
        last_logits = logits[:, -1, pref_ids].detach()
        probs = torch.softmax(last_logits, dim=-1)
        scores = (probs @ self._weights.to(probs.dtype)).tolist()
        return [float(s) for s in scores]

    def score_and_comment(self, img: Image.Image) -> Tuple[Dict[str, float], float, Dict[str, str]]:
        """对图像进行 7 维度打分与评语，并返回总分。"""
        pixel_values = to_model_tensor(img).to(torch.bfloat16).to(self.device)

        # 7 维度分数（重用数字→字母映射提示，输出 2 字母）
        base_suffix = (
            "Rate the score of the image in 0-100. In the output format, numbers are replaced by 2 corresponding "
            "letters with mapping: 0-aa ... 100-ey. The answer only outputs 2 corresponding letters."
        )
        questions = [f"Please evaluate {name}. {base_suffix}" for name in AESTHETIC_DIMENSIONS]
        aspect_scores_list = self._score_questions(questions, pixel_values)
        aspect_scores = {n: float(v) for n, v in zip(AESTHETIC_DIMENSIONS, aspect_scores_list)}

        # 总分使用官方 score 接口
        gen_cfg = dict(self.generation_config)
        total_score = float(self.model.score(self.device, self.tokenizer, pixel_values, gen_cfg))

        # 维度评语（批量）
        prompts = [
            f"Please evaluate the aesthetic quality of this image from the aspect of {name}."
            for name in AESTHETIC_DIMENSIONS
        ]
        num_patches_list = [pixel_values.size(0)] * len(prompts)
        pixel_values_batch = torch.cat((pixel_values,) * len(prompts), dim=0)
        comments = self.model.batch_chat(
            self.device,
            self.tokenizer,
            pixel_values_batch,
            questions=prompts,
            generation_config=self.generation_config,
            num_patches_list=num_patches_list,
        )
        aspect_comments = {n: c for n, c in zip(AESTHETIC_DIMENSIONS, comments)}
        return aspect_scores, total_score, aspect_comments

