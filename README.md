# USIR Inference

## Environment

```bash
conda env create -f environment.yaml
conda activate usir-public
```

## Weights

Download the FLUX.2-klein-base-9B base model and provide its local path with
`--model_path`.

Download the USIR adapter from [anonymous1usir/UISR_adapter](https://huggingface.co/anonymous1usir/UISR_adapter) and place these files at:

```text
weights/adapter/adapter_model.safetensors
weights/adapter/adapter_config.json
```

## Run

Single image:

```bash
python infer.py \
  --model_path /path/to/FLUX.2-klein-base-9B \
  --input_path examples/input.png \
  --output_dir outputs/single
```

Directory:

```bash
python infer.py \
  --model_path /path/to/FLUX.2-klein-base-9B \
  --input_path /path/to/images \
  --output_dir outputs/restored
```
