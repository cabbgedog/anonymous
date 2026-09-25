"""USIR fixed-prompt, nonprivileged image restoration inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from usir_inference.embedding import load_fixed_embeddings
from usir_inference.flux_ops import load_runtime, restore
from usir_inference.image_ops import discover_images, resize_long_edge

DEFAULT_SIGMA = 0.43612543


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_image(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    array = ((tensor[0].detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
    image = Image.fromarray(array.permute(1, 2, 0).numpy().round().astype(np.uint8))
    return image.crop((0, 0, *size))


def pad_to_multiple(image: Image.Image, multiple: int = 16) -> Image.Image:
    width, height = image.size
    padded = Image.new("RGB", ((width + multiple - 1) // multiple * multiple, (height + multiple - 1) // multiple * multiple))
    padded.paste(image, (0, 0))
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="USIR fixed-prompt nonprivileged inference without a text encoder.")
    parser.add_argument("--model_path", type=Path, required=True, help="Official FLUX.2-klein-base-9B directory.")
    parser.add_argument("--adapter_path", type=Path, default=Path("weights/adapter"))
    parser.add_argument("--embedding_path", type=Path, default=Path("weights/fixed_prompt_embeddings.safetensors"))
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--main_sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--max_long_edge", type=int, default=1024)
    args = parser.parse_args()
    samples = discover_images(args.input_path)
    if not samples:
        raise ValueError("no supported images found")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_embeds, txt_ids = load_fixed_embeddings(args.embedding_path)
    pipe = load_runtime(str(args.model_path), str(args.adapter_path), device)
    manifest = args.output_dir / "run_manifest.jsonl"
    with torch.inference_mode(), manifest.open("w", encoding="utf-8") as handle:
        for index, (input_path, relative_path) in enumerate(samples):
            with Image.open(input_path) as source:
                original = resize_long_edge(source.convert("RGB"), args.max_long_edge)
            content_size = original.size
            restored = restore(pipe, image_tensor(pad_to_multiple(original)).to(device=device, dtype=torch.bfloat16), prompt_embeds, txt_ids, seed=args.seed + index, main_sigma=args.main_sigma)
            output_path = args.output_dir / relative_path.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_image(restored, content_size).save(output_path)
            handle.write(json.dumps({"input": relative_path.as_posix(), "output": output_path.relative_to(args.output_dir).as_posix(), "seed": args.seed + index, "main_sigma": args.main_sigma, "fixed_prompt_embeddings": args.embedding_path.name}) + "\n")
            print(f"[{index + 1}/{len(samples)}] {relative_path}", flush=True)


if __name__ == "__main__":
    main()
