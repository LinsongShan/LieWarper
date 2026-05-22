import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
import numpy as np
import math
from typing import Tuple, Optional, Union, Dict, Any, List
import types

from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    ext_freqs=None,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    if ext_freqs is None:
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
            / linear_factor
        )
    else:
        freqs = ext_freqs
    freqs = torch.outer(pos, freqs)
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class CustomWanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        u,
        v,
        ext_freqs_h=None,
        ext_freqs_w=None,
        theta: float = 10000.0,
        enable_smoothing: bool = True,
        enable_temporal_smoothing: bool = False,
        divisor: int = 16,
        mu: int = 21,
        std: int = 11,
        mu_time: int = 11,
        std_time: int = 5,
        enable_lookup_norm: bool = False,
        lookup_norm_thr: float = 1.0,
        num_frames: int = 13,
        H_lat: int = 60,
        W_lat: int = 104,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        p_h = p_w = 2
        H = H_lat // p_h
        W = W_lat // p_w

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        def _gaussian_blur(x, k=5, sigma=3.0):
            kernel_1d = torch.tensor(
                [math.exp(-((i - (k // 2)) ** 2) / (2 * sigma**2)) for i in range(k)], device=x.device, dtype=x.dtype
            )
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d[None, None, :, :].repeat(1, 1, 1, 1)
            return conv2d(x.unsqueeze(1), kernel_2d, padding=k // 2).squeeze(1)

        if enable_smoothing:
            u_ds = F.interpolate(u.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            v_ds = F.interpolate(v.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

            u_ds = _gaussian_blur(u_ds, k=mu, sigma=std)
            v_ds = _gaussian_blur(v_ds, k=mu, sigma=std)

            if enable_temporal_smoothing:
                kernel_size = mu_time
                sigma_t = std_time
                kernel_t = torch.tensor(
                    [math.exp(-((i - (kernel_size // 2)) ** 2) / (2 * sigma_t**2)) for i in range(kernel_size)],
                    device=u_ds.device,
                    dtype=u_ds.dtype,
                )
                kernel_t = kernel_t / kernel_t.sum()

                u_t = u_ds.unsqueeze(0).unsqueeze(0)
                v_t = v_ds.unsqueeze(0).unsqueeze(0)

                u_smooth = F.conv3d(u_t, kernel_t.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))
                v_smooth = F.conv3d(v_t, kernel_t.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))

                u_lat = u_smooth.squeeze(0).squeeze(0) / divisor
                v_lat = v_smooth.squeeze(0).squeeze(0) / divisor
            else:
                u_lat = u_ds / divisor
                v_lat = v_ds / divisor
        else:
            u_lat = (
                F.interpolate(u.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1) / divisor
            )
            v_lat = (
                F.interpolate(v.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1) / divisor
            )

        cum_u, cum_v = [], []
        u_c = torch.zeros_like(u_lat[0])
        v_c = torch.zeros_like(v_lat[0])

        if enable_lookup_norm:
            threshold = lookup_norm_thr
            eps = 1e-6
            cum_u, cum_v = [], []
            u_c = torch.zeros_like(u_lat[0])
            v_c = torch.zeros_like(v_lat[0])

            for t in range(u_lat.shape[0]):
                if t < u_lat.shape[0] - 1:
                    du = (u_lat[t + 1] - u_lat[t]).norm()
                    dv = (v_lat[t + 1] - v_lat[t]).norm()
                    diff = torch.sqrt(du * du + dv * dv)
                else:
                    diff = torch.tensor(0.0, dtype=u_lat.dtype, device=u_lat.device)

                w = torch.clamp(threshold / (diff + eps), max=1.0)

                u_c = u_c + w * u_lat[t]
                v_c = v_c + w * v_lat[t]
                cum_u.append(u_c.clone())
                cum_v.append(v_c.clone())

            cum_u = torch.stack(cum_u)
            cum_v = torch.stack(cum_v)
        else:
            for t in range(u_lat.shape[0]):
                u_c = u_c + u_lat[t]
                v_c = v_c + v_lat[t]
                cum_u.append(u_c.clone())
                cum_v.append(v_c.clone())
            cum_u = torch.stack(cum_u)
            cum_v = torch.stack(cum_v)

        grid_h_custom, grid_w_custom = torch.meshgrid(
            torch.arange(H, device=u.device), torch.arange(W, device=u.device), indexing="ij"
        )
        h_motion = grid_h_custom[None] + cum_v
        w_motion = grid_w_custom[None] + cum_u

        h_motion_reshaped = h_motion.permute(0, 2, 1).reshape(-1, H)
        w_motion_reshaped = w_motion.permute(0, 1, 2).reshape(-1, W)

        self.orig_freq_t = get_1d_rotary_pos_embed(
            dim=t_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )
        self.orig_freq_h = get_1d_rotary_pos_embed(
            dim=h_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )
        self.orig_freq_w = get_1d_rotary_pos_embed(
            dim=w_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )

        h_list = []
        w_list = []

        for i in range(h_motion_reshaped.size(0)):
            if ext_freqs_h is not None:
                ext_freqs = ext_freqs_h[i]
            else:
                ext_freqs = None
            h_i = get_1d_rotary_pos_embed(
                dim=h_dim,
                pos=h_motion_reshaped[i],
                theta=theta,
                ext_freqs=ext_freqs,
                use_real=False,
                repeat_interleave_real=False,
            )
            h_list.append(h_i)

        for i in range(w_motion_reshaped.size(0)):
            if ext_freqs_w is not None:
                ext_freqs = ext_freqs_w[i]
            else:
                ext_freqs = None
            w_i = get_1d_rotary_pos_embed(
                dim=w_dim,
                pos=w_motion_reshaped[i],
                theta=theta,
                ext_freqs=ext_freqs,
                use_real=False,
                repeat_interleave_real=False,
            )
            w_list.append(w_i)

        self.h_freqs = torch.stack(h_list).reshape(num_frames, W, H, -1).permute(0, 2, 1, 3)
        self.w_freqs = torch.stack(w_list).reshape(num_frames, H, W, -1)
        self.t_freqs = get_1d_rotary_pos_embed(
            t_dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs_f = self.t_freqs[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = self.h_freqs
        freqs_w = self.w_freqs

        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class CustomWanAttnProcessor2_0(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        custom_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if custom_rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, custom_rotary_emb)
            key = apply_rotary_emb(key, custom_rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomWanTransformerBlock(WanTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        custom_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb, custom_rotary_emb=custom_rotary_emb
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


def custom_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    rotary_emb: Optional[torch.Tensor] = None,
    custom_rotary_emb: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            print("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block_idx, block in enumerate(self.blocks):
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, custom_rotary_emb
            )
    else:
        for block_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, custom_rotary_emb)

    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class WanModuleUtils:
    @staticmethod
    def modify_transformer_forward(model):
        model.forward = types.MethodType(custom_transformer_forward, model)
        return model

    @staticmethod
    def modify_transformer_layers(model):
        config = model.config
        num_attention_heads = config.num_attention_heads
        attention_head_dim = config.attention_head_dim
        ffn_dim = config.ffn_dim
        qk_norm = config.qk_norm
        cross_attn_norm = config.cross_attn_norm
        eps = config.eps
        added_kv_proj_dim = config.added_kv_proj_dim
        num_layers = config.num_layers
        inner_dim = num_attention_heads * attention_head_dim

        for idx in range(num_layers):
            b = CustomWanTransformerBlock(
                inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
            )
            b.load_state_dict(model.blocks[idx].state_dict(), strict=True)
            model.blocks[idx] = b.to(device=model.device, dtype=model.dtype)

        for out_i in range(num_layers):
            processor = CustomWanAttnProcessor2_0()
            model.blocks[out_i].attn1.set_processor(processor)

        return model