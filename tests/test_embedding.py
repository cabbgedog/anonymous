import torch
from safetensors.torch import save_file

from usir_inference.embedding import load_fixed_embeddings


def test_load_fixed_embeddings_requires_two_tensors(tmp_path):
    path = tmp_path / "bad.safetensors"
    save_file({"prompt_embeds": torch.zeros(1)}, path)
    try:
        load_fixed_embeddings(path)
    except ValueError as error:
        assert "txt_ids" in str(error)
    else:
        raise AssertionError("missing txt_ids must fail")
