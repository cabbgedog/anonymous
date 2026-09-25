"""Flux runtime operations for fixed-embedding USIR restoration."""

import torch
from diffusers import Flux2KleinPipeline


def load_runtime(model_path: str, adapter_path: str, device: torch.device) -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.load_lora_weights(adapter_path, weight_name="adapter_model.safetensors")
    pipe.to(device)
    pipe.vae.eval()
    pipe.transformer.eval()
    return pipe


@torch.no_grad()
def restore(
    pipe: Flux2KleinPipeline,
    pixels: torch.Tensor,
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    *,
    seed: int,
    main_sigma: float,
) -> torch.Tensor:
    device = pixels.device
    generator = torch.Generator(device=device).manual_seed(seed)
    spatial = pipe._encode_vae_image(pixels, generator=generator)
    latents = pipe._pack_latents(spatial)
    latent_ids = pipe._prepare_latent_ids(spatial).to(device)
    condition_ids = latent_ids.clone()
    condition_ids[:, :, 0] = 10
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    sigma = torch.tensor([main_sigma], device=device, dtype=torch.float32)
    sigma_latents = sigma.to(latents.dtype).view(-1, 1, 1)
    main_latents = (1.0 - sigma_latents) * latents + sigma_latents * noise
    condition_latents = latents
    velocity = pipe.transformer(
        hidden_states=torch.cat([main_latents, condition_latents], dim=1),
        timestep=sigma,
        guidance=None,
        encoder_hidden_states=prompt_embeds.to(device=device, dtype=torch.bfloat16),
        txt_ids=txt_ids.to(device=device),
        img_ids=torch.cat([latent_ids, condition_ids], dim=1),
        return_dict=False,
    )[0][:, : latents.shape[1]]
    endpoint = main_latents - sigma_latents * velocity
    unpacked = pipe._unpack_latents_with_ids(endpoint, latent_ids, height=spatial.shape[-2], width=spatial.shape[-1])
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype) + pipe.vae.config.batch_norm_eps)
    return pipe.vae.decode(pipe._unpatchify_latents(unpacked * std + mean), return_dict=False)[0]
