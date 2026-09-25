# USIR: Fixed-Prompt Image Restoration Inference

This release provides inference for the USIR non-privileged adapter. It uses a
precomputed embedding of the fixed restoration instruction and therefore does
**not** load or run a text encoder at inference time.

## Requirements

Create the environment:

```bash
conda env create -f environment.yaml
conda activate usir-public
```

Obtain the official `FLUX.2-klein-base-9B` base model separately and comply
with its upstream license. The base model is intentionally not redistributed
in this repository. Place it locally and pass its path with `--model_path`.

Download the USIR LoRA adapter from the anonymous Hugging Face release and
place its files in `weights/adapter/`:

```text
weights/adapter/adapter_model.safetensors
weights/adapter/adapter_config.json
```

The adapter is intentionally excluded from this GitHub repository. The fixed
prompt embedding file remains included under `weights/`.

## Inference

Single image:

```bash
python infer.py \
  --model_path /path/to/FLUX.2-klein-base-9B \
  --input_path examples/input.png \
  --output_dir outputs/single
```

Directory input:

```bash
python infer.py \
  --model_path /path/to/FLUX.2-klein-base-9B \
  --input_path /path/to/images \
  --output_dir outputs/restored \
  --seed 42 --main_sigma 0.43612543 --max_long_edge 1024
```

The output directory must be empty. Recursive input paths are retained, all
outputs are PNG files, and `run_manifest.jsonl` records the relative file name,
seed, and fixed inference parameters. A modern CUDA GPU with approximately
16 GB or more VRAM is recommended.

## Fixed-prompt protocol

This package supports only the non-privileged clean-restoration protocol. It
uses `sigma_c=0`, `main_sigma=0.43612543`, and the saved tensors in
`weights/fixed_prompt_embeddings.safetensors`. Custom text prompts are not
supported because the runtime intentionally excludes the text encoder.

## License

The code in this repository is released under MIT. The adapter and external
base model remain subject to their applicable model licenses.
