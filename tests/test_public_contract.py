from pathlib import Path


def test_runtime_source_does_not_encode_or_load_text_encoder():
    source = Path("infer.py").read_text(encoding="utf-8") + Path("usir_inference/flux_ops.py").read_text(encoding="utf-8")
    assert "encode_prompt(" not in source
    assert ".text_encoder" not in source
