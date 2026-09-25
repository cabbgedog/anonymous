"""Load the precomputed fixed prompt tensors used by public inference."""

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_fixed_embeddings(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_file(str(path), device="cpu")
    missing = {"prompt_embeds", "txt_ids"} - set(tensors)
    if missing:
        raise ValueError(f"fixed embedding file is missing: {', '.join(sorted(missing))}")
    return tensors["prompt_embeds"], tensors["txt_ids"]
