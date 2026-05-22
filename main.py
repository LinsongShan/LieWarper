import argparse
import json
from typing import Tuple, Optional, Union, List, Any, Dict
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
import os
import numpy as np
from tqdm import tqdm
import gc
import decord
import random
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel, T5EncoderModel, T5Tokenizer
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from lie_utils import (
    Lie_Basis, 
    apply_tensor_low_rank_filter, 
    estimate_global_lie_params, 
    apply_lie_affine_warp
)
from rope_utils import CustomWanRotaryPosEmbed, WanModuleUtils

decord.bridge.set_bridge("torch")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_rgb_flow(of_model, video_frames, flow2img=True):
    flow_feats = []
    for i in range(len(video_frames) - 1):
        flow = of_model(video_frames[i : i + 1], video_frames[i + 1 : i + 2])[-1]
        flow_feats.append(flow)
    flow_feats = torch.cat(flow_feats, dim=0)
    if flow2img:
        flow_feats = flow_to_image(flow_feats).to(video_frames.dtype) / 127.5 - 1
    return flow_feats


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif frame_sample_step is not None:
            indices = list(range(start_frame, end_frame, frame_sample_step))
        else:
            indices = np.linspace(start_frame, end_frame - 1, max_num_frames).astype(int).tolist()

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()

        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()


def encode_video(vae, video, device):
    if video.dim() == 4:
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    elif video.shape[0] == 3:
        video = video.unsqueeze(0)
    elif video.dim() == 5:
        video = video.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")

    video = video.to(device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latents = latent_dist.sample()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    return latents.to(memory_format=torch.contiguous_format).float()


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, num_videos_per_prompt=1, max_sequence_length=226, device=None, dtype=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds


def compute_prompt_embeddings(tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad=False):
    if requires_grad:
        prompt_embeds = _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, 1, max_sequence_length, device, dtype)
    else:
        with torch.no_grad():
            prompt_embeds = _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, 1, max_sequence_length, device, dtype)
    return prompt_embeds


def phase_constraint_loss(pred, tgt, eps: float = 1e-8, norm: str = "ortho") -> torch.Tensor:
    original_dtype = pred.dtype
    pred = pred.to(torch.float32)
    tgt = tgt.to(torch.float32)
    
    Fp = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm=norm)
    Ft = torch.fft.rfftn(tgt, dim=(-3, -2, -1), norm=norm)

    zp = Fp / (Fp.abs() + eps)
    zt = Ft / (Ft.abs() + eps)

    loss = F.l1_loss(zp.real, zt.real) + F.l1_loss(zp.imag, zt.imag)
    return loss.to(original_dtype)


def tune_p(
    ## Models
    vae,
    transformer,
    tokenizer,
    text_encoder,
    scheduler,
    ## Inputs
    video_path: str,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    ## Constants
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    ## Generation
    latents: Optional[torch.Tensor] = None,
    num_frames: int = 81,
    mse_weight=1.0,
    guidance_scale: float = 5.0,
    ## RoPE Warping Args
    train_mode: bool = True,
    divisor: int = 16,
    enable_smoothing: bool = True,
    mu: int = 21,
    std: int = 11,
    mu_time: int = 21,
    std_time: int = 11,
    enable_lookup_norm: bool = False,
    lookup_norm_thr: float = 1.0,
    n_replace_gt_mod: list = [],
    num_optim_steps: int = 5,
    tune_wo_warped_uv: bool = False,
    theta: float = 10000.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    ## Others
    output_type: Optional[str] = "np",
    return_dict: bool = True,
):
    weights = Raft_Small_Weights.DEFAULT
    of_model = raft_small(weights=weights).to("cuda")
    of_model.eval()

    skip_frames_start = 0
    skip_frames_end = 0
    max_num_frames = num_frames
    frame_sample_step = None

    prompt_embeds = compute_prompt_embeddings(
        tokenizer, text_encoder, prompt, 226, "cuda", torch.float32, requires_grad=False
    ).to("cuda")
    
    negative_prompt_embeds = compute_prompt_embeddings(
        tokenizer, text_encoder, negative_prompt, 226, "cuda", torch.float32, requires_grad=False
    ).to("cuda")

    transformer_dtype = transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    scheduler.set_timesteps(num_inference_steps, device="cuda")
    timesteps = scheduler.timesteps

    with torch.no_grad():
        print("Reading reference video...")
        video_frames = get_video_frames(
            video_path=video_path,
            width=width,
            height=height,
            skip_frames_start=skip_frames_start,
            skip_frames_end=skip_frames_end,
            max_num_frames=max_num_frames,
            frame_sample_step=frame_sample_step,
        ).to(device="cuda")
        vid_indices = torch.linspace(0, video_frames.size(0) - 1, num_frames).long()
        video_frames = video_frames[vid_indices]
        
        latent_video = encode_video(vae=vae, video=video_frames, device="cuda")
        
        gt_flow_feats = extract_rgb_flow(of_model=of_model, video_frames=video_frames, flow2img=False)
        gt_flow_feats = gt_flow_feats.to(device="cuda", dtype=torch.float32)

        indices = torch.linspace(0, gt_flow_feats.size(0) - 1, 13 if num_frames == 49 else 21).long()
        gt_flow_feats = gt_flow_feats[indices]
        
        H_lat = height // 8
        W_lat = width // 8
        
        custom_rope = CustomWanRotaryPosEmbed(
            attention_head_dim=transformer.config.attention_head_dim,
            patch_size=transformer.config.patch_size,
            max_seq_len=transformer.config.rope_max_seq_len,
            u=gt_flow_feats[:, 0, :, :].to("cpu"),
            v=gt_flow_feats[:, 1, :, :].to("cpu"),
            divisor=divisor,
            enable_smoothing=enable_smoothing,
            theta=theta,
            mu=mu,
            std=std,
            mu_time=mu_time,
            std_time=std_time,
            enable_lookup_norm=enable_lookup_norm,
            lookup_norm_thr=lookup_norm_thr,
            H_lat=H_lat,
            W_lat=W_lat,
            num_frames=13 if num_frames == 49 else 21,
        )
        custom_image_rotary_emb = custom_rope(latents).to(device="cuda")
        gt_image_rotary_emb = transformer.rope(latents).to(device="cuda")

        if tune_wo_warped_uv:
            custom_image_rotary_emb = gt_image_rotary_emb

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    of_model.requires_grad_(False)
    del text_encoder
    del of_model
    clean_memory()

    video_processor = VideoProcessor(vae_scale_factor=8)
    
    # ================= Initialization =================
    if n_replace_gt_mod:
        if train_mode:
            tunable_phase_shift = [
                torch.nn.Parameter(torch.zeros_like(custom_image_rotary_emb, dtype=torch.float32))
                for _ in range(len(n_replace_gt_mod))
            ]

            print("Initializing Camera Trajectory via Tensor Decomposition...")
            H_flow, W_flow = gt_flow_feats.shape[2], gt_flow_feats.shape[3]
            filtered_flow_for_solver = apply_tensor_low_rank_filter(gt_flow_feats, rank=3)
            init_xi = estimate_global_lie_params(filtered_flow_for_solver, H_flow, W_flow)
            scale_factor = float(gt_flow_feats.shape[0]) 
            lie_params = torch.nn.Parameter((init_xi * scale_factor).clone().detach().requires_grad_(True))
            
            optimizer = torch.optim.AdamW([
                {'params': tunable_phase_shift, 'lr': 1e-2},
                {'params': [lie_params], 'lr': 1e-3}
            ])
        else:
            tunable_phase_shift = [torch.zeros_like(custom_image_rotary_emb, dtype=torch.float32) for _ in range(len(n_replace_gt_mod))]
            lie_params = torch.zeros(4, device="cuda", dtype=torch.float32)
    # =================================================

    for i, t in tqdm(enumerate(timesteps), desc="Inference", total=len(timesteps)):
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0]).to("cuda")
        current_optimized_rope = None
        patch_size = transformer.config.patch_size
        
        rope_F = latents.shape[2] // patch_size[0]
        rope_H = latents.shape[3] // patch_size[1]
        rope_W = latents.shape[4] // patch_size[2]
        rope_shape_info = (rope_F, rope_H, rope_W)

        if i in n_replace_gt_mod:
            if train_mode:
                sigmas = scheduler.sigmas.to("cuda")[i]
                while sigmas.ndim < latent_model_input.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                for optim_iter in range(num_optim_steps):
                    raw_delta_theta = sum(tunable_phase_shift[: i + 1])
                    warped_delta_theta = apply_lie_affine_warp(
                        raw_delta_theta, lie_params, Lie_Basis, rope_shape_info
                    )
                    rotation_operator = torch.complex(torch.cos(warped_delta_theta), torch.sin(warped_delta_theta))
                    optimized_rope = custom_image_rotary_emb * rotation_operator

                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        custom_rotary_emb=optimized_rope,
                    )[0]

                    target_velocity = (latent_model_input - latent_video.to(noise_pred.dtype)) / sigmas
                    target_velocity = target_velocity.to(noise_pred.dtype)
                    mse_val = torch.nn.functional.mse_loss(noise_pred, target_velocity, reduction="mean")
                    loss_phase = phase_constraint_loss(noise_pred, target_velocity)
                    
                    loss = loss_phase + mse_weight * mse_val
                    if loss.dtype != noise_pred.dtype:
                        loss = loss.to(noise_pred.dtype)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                      raw_delta_theta = sum(tunable_phase_shift[: i + 1])
                      warped_delta_theta = apply_lie_affine_warp(
                        raw_delta_theta, lie_params, Lie_Basis, rope_shape_info
                      )
                      rotation_operator = torch.complex(torch.cos(warped_delta_theta), torch.sin(warped_delta_theta))
                      current_optimized_rope = custom_image_rotary_emb * rotation_operator

            with torch.no_grad():      
                if current_optimized_rope is None:
                      raw_delta_theta = sum(tunable_phase_shift[: i + 1])
                      warped_delta_theta = apply_lie_affine_warp(
                        raw_delta_theta, lie_params, Lie_Basis, rope_shape_info
                      )
                      rotation_operator = torch.complex(torch.cos(warped_delta_theta), torch.sin(warped_delta_theta))
                      current_optimized_rope = custom_image_rotary_emb * rotation_operator

                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=current_optimized_rope,
                )[0]

                noise_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=current_optimized_rope,
                )[0]

                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        else:
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=gt_image_rotary_emb,
                )[0]

                noise_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=gt_image_rotary_emb,
                )[0]

                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
    if not output_type == "latent":
        latents = latents.to(vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = vae.decode(latents, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    if not return_dict:
        return (video,)

    return WanPipelineOutput(frames=video)


# Global Configs
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832
NUM_INFERENCE_STEPS = 50
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# Model Loading
print("Loading models...")
scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32,).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to("cuda")
transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")

transformer = WanModuleUtils.modify_transformer_layers(transformer)
transformer = WanModuleUtils.modify_transformer_forward(transformer)
transformer.enable_gradient_checkpointing()

seed_everything(2026)
H_lat = VIDEO_HEIGHT // 8
W_lat = VIDEO_WIDTH // 8
latents_init = torch.randn(1, 16, 13, H_lat, W_lat).to("cuda")


def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    MSE_WEIGHT = 0.1
    NUM_OPTIM_STEPS = 5
    N_REPLACE_GT_MOD = 10
    START_WITH_UV_WARPED = 1 

    output = tune_p(
        ## Models
        vae=vae,
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        ## Inputs
        video_path=args.input_video,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        ## Constants
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        ## Generation
        latents=latents_init.clone(),
        num_frames=args.frames,        
        mse_weight=MSE_WEIGHT,
        guidance_scale=5.0,
        ## RoPE Warping Args
        train_mode=True,
        divisor=8, 
        enable_smoothing=True,
        mu=21,
        std=11, 
        mu_time=5,
        std_time=3,
        enable_lookup_norm=True,
        lookup_norm_thr=1.0,
        n_replace_gt_mod=range(N_REPLACE_GT_MOD),
        num_optim_steps=NUM_OPTIM_STEPS,
        tune_wo_warped_uv=(1 - START_WITH_UV_WARPED),
    ).frames[0]

    filename = f"{prompt_clean(args.prompt).replace(' ', '_')[:20]}.mp4"
    save_path = os.path.join(args.output_dir, filename)
    export_to_video(output, save_path, fps=15)
    print(f"Video saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.1 Video Generation with Lie Algebra Trajectory Refinement")
    parser.add_argument(
        "--input_video",
        type=str,
        default="./input_video.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat playing with a ball of yarn",
        help="Text prompt to guide the video generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=49,
        choices=[49, 81],
        help="Number of frames to process",
    )

    args = parser.parse_args()
    print("Arguments:", args)

    generate(args)