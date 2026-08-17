# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 model implementation.

This module implements the RND1 architecture with bidirectional attention for
diffusion-based language modeling, with Mixture of Experts (MoE) support via
transformers' Qwen3MoeExperts and Qwen3MoeTopKRouter.

Based on the Qwen3Moe architecture:
https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
"""

from __future__ import annotations

import os

import torch

from torch import nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import MaskedLMOutput, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

from .configuration_rnd import RND1Config
from .generation_utils import RND1GenerationMixin

# Register rnd1 to use the same checkpoint weight conversion as qwen2_moe.
# This enables automatic fusion of per-expert weights (experts.N.{gate,up,down}_proj)
# into the 3D tensor format (experts.{gate_up_proj, down_proj}) used by Qwen3MoeExperts.
register_checkpoint_conversion_mapping(
    "rnd1", get_checkpoint_conversion_mapping("qwen2_moe"), overwrite=True
)

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match query heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RND1Attention(nn.Module):
    """RND1 attention layer with bidirectional attention for diffusion modeling."""

    def __init__(self, config: RND1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = getattr(config, "sliding_window", None)

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        dual_cache: bool | None = False,
        replace_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        use_sdpa = getattr(self.config, "_attn_implementation", "eager") == "sdpa"

        if use_sdpa:  # noqa: SIM102
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):  # noqa: SIM102
                if attention_mask.dtype not in [
                    torch.bool,
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ]:
                    attention_mask = attention_mask.to(dtype=query_states.dtype)

            assert not self.is_causal, f"Attention layer {self.layer_idx} is causal"
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(bsz, q_len, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)
            return attn_out, None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_out = torch.matmul(attn_weights, value_states)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(hidden_states.size(0), hidden_states.size(1), -1)
        )
        attn_out = self.o_proj(attn_out)

        return attn_out, None


class RND1SparseMoeBlock(nn.Module):
    """RND1 Sparse MoE block using fused 3D-tensor experts for fast kernel dispatch.

    Uses Qwen3MoeExperts (decorated with @use_experts_implementation) which stores
    all expert weights as stacked 3D tensors, enabling optimized grouped-GEMM kernels
    when available.
    """

    def __init__(self, config: RND1Config):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class RND1DecoderLayer(nn.Module):
    """RND1 decoder layer with bidirectional attention for diffusion language modeling."""

    def __init__(self, config: RND1Config, layer_idx: int):
        super().__init__()
        self.self_attn = RND1Attention(config, layer_idx)
        self.mlp = RND1SparseMoeBlock(config)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        replace_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights


class RND1PreTrainedModel(PreTrainedModel):
    """Base class for RND1 models with weight initialization and loading support."""

    config_class = RND1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RND1DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # No-op: weights are loaded from pretrained checkpoint
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ):
        """Load pretrained model with generation config."""
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )

        resume_download = kwargs.get("resume_download")
        proxies = kwargs.get("proxies")
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline")

        _model.generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )

        # Re-compute rotary embedding inv_freq buffers (non-persistent, not in checkpoint).
        # With low_cpu_mem_usage=True, these are created on meta device during __init__
        # and stay as zeros after weight loading since they're not in the safetensors file.
        model_core = getattr(_model, "model", _model)
        device = next(_model.parameters()).device
        for module in model_core.modules():
            if isinstance(module, Qwen3MoeRotaryEmbedding) and hasattr(module, "inv_freq"):
                inv_freq, attention_scaling = (
                    Qwen3MoeRotaryEmbedding.compute_default_rope_parameters(_model.config, device)
                )
                module.inv_freq = inv_freq
                module.attention_scaling = attention_scaling
                module.original_inv_freq = inv_freq

        return _model


class RND1Model(RND1PreTrainedModel):
    """RND1 transformer model with bidirectional attention for diffusion language modeling."""

    def __init__(self, config: RND1Config):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RND1DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        """Forward pass through the RND1 model."""

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            router_logits=None,
        )


class RND1LM(RND1PreTrainedModel, RND1GenerationMixin):
    """Radical Numerics Diffusion Language Model with bidirectional attention."""

    def __init__(self, config: RND1Config):
        super().__init__(config)
        self.model = RND1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Get the output embeddings layer (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings layer (lm_head)."""
        self.lm_head = new_embeddings

    @classmethod
    def can_generate(cls) -> bool:
        """Indicates this model can generate text."""
        return True

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> MaskedLMOutput:
        """Forward pass with optional loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )
