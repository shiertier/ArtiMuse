import os
import json
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

import sys
sys.path.append("src")
sys.path.append("src/artimuse")
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

# Aesthetic Attributes
AESTHETIC_ATTRIBUTES = [
    "Composition & Design",
    "Visual Elements & Structure",
    "Technical Execution",
    "Originality & Creativity",
    "Theme & Communication",
    "Emotion & Viewer Response",
    "Overall Gestalt",
    "Comprehensive Evaluation",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)  


def main(args):
    # Setup paths
    model_path = os.path.join("checkpoints", args.model_name)
    results_dir = os.path.join("results", "image_results")
    os.makedirs(results_dir, exist_ok=True)

    # Load model & tokenizer
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    generation_config = dict(max_new_tokens=8192, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    # Load image
    pixel_values = load_image(args.image_path).to(torch.bfloat16).to(args.device)

    # Inference
    results = {}
    score = model.score(args.device, tokenizer, pixel_values, generation_config)
    results["Aesthetic Score"] = score

    results["Aesthetic Attributes"] = {}
    for aspect in AESTHETIC_ATTRIBUTES:
        prompt = f"Please evaluate the aesthetic quality of this image from the aspect of {aspect}."
        response = model.chat(args.device, tokenizer, pixel_values, prompt, generation_config)
        results["Aesthetic Attributes"][aspect] = response

    # Save results
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(results_dir, f"{image_name}_{args.model_name}_eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ArtiMuse", help="Name of the model (must exist in checkpoints/)")
    parser.add_argument("--image_path", type=str, default="example/test.jpg", help="Path to test image")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device number")
    args = parser.parse_args()

    main(args)
