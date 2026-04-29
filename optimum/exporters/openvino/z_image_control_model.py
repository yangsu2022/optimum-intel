# -*- coding: utf-8 -*-
# Copyright (c) 2026 optimum-intel contributors
# OV-friendly ZImage Control Transformer for OpenVINO export
#
# This module provides an OV-exportable version of VideoX-Fun's ZImageControlTransformer2DModel.
# It inherits from diffusers' ZImageTransformer2DModel and adds control branches that are
# compatible with OpenVINO's torch.jit.trace-based export.

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
    RMSNorm,
    FinalLayer,
    RopeEmbedder,
    ADALN_EMBED_DIM,
)
from diffusers.configuration_utils import register_to_config


SEQ_MULTI_OF = 32


class OVZImageControlTransformerBlock(nn.Module):
    """
    OV-friendly version of ZImageControlTransformerBlock.
    
    This block processes control context and outputs hints to be added to the main transformer.
    The key difference from VideoX-Fun's version is that it uses static operations
    compatible with torch.jit.trace.
    """
    
    def __init__(
        self, 
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        block_id: int = 0
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.block_id = block_id
        self.modulation = modulation
        
        # Create the base transformer block components
        # Attention
        from diffusers.models.attention_processor import Attention
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(
            query_dim=dim,
            heads=n_heads,
            kv_heads=n_kv_heads,
            dim_head=dim // n_heads,
            bias=False,
            out_bias=True,
            qk_norm="rms_norm" if qk_norm else None,
        )
        
        # FFN
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        
        # Modulation (if enabled)
        if modulation:
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )
        
        # Control-specific layers
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        else:
            self.before_proj = None
            
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
    
    def forward(
        self,
        c: torch.Tensor,  # Control context or stacked hints+c
        x: torch.Tensor,  # Main hidden states for block_id==0
        attn_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        adaln_input: Optional[torch.Tensor] = None,
        prev_hints: Optional[torch.Tensor] = None,  # [num_hints, B, S, D] for block_id > 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns (updated_hints_stack, current_c).
        
        For block_id == 0: c is control context, uses before_proj(c) + x
        For block_id > 0: prev_hints contains previous hints, c is last element
        
        Returns:
            hints_stack: [num_hints+1, B, S, D] - all hints including new one
            c: [B, S, D] - updated control context
        """
        if self.block_id == 0:
            c = self.before_proj(c) + x
            # No previous hints
            num_prev_hints = 0
        else:
            # c is the last element, prev_hints contains all previous hints
            num_prev_hints = prev_hints.shape[0] if prev_hints is not None else 0
        
        # Apply modulation if enabled
        if self.modulation and adaln_input is not None:
            adaln_out = self.adaln(adaln_input)
            shift1, scale1, shift2, scale2 = adaln_out.chunk(4, dim=-1)
            
            # Attention with modulation
            c_normed = self.attention_norm1(c) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            c = c + self.attention(c_normed, attention_mask=attn_mask, freqs_cis=freqs_cis)
            
            # FFN with modulation
            c_ffn = self.ffn_norm1(c) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            c = c + self.ffn_norm2(self.feed_forward(c_ffn))
        else:
            # Without modulation
            c = c + self.attention(self.attention_norm1(c), attention_mask=attn_mask, freqs_cis=freqs_cis)
            c = c + self.ffn_norm2(self.feed_forward(self.ffn_norm1(c)))
        
        # Generate hint from this block
        c_skip = self.after_proj(c)
        
        # Stack hints: [prev_hints..., c_skip]
        if self.block_id == 0:
            hints_stack = c_skip.unsqueeze(0)  # [1, B, S, D]
        else:
            hints_stack = torch.cat([prev_hints, c_skip.unsqueeze(0)], dim=0)
        
        return hints_stack, c


class OVBaseZImageTransformerBlock(nn.Module):
    """
    OV-friendly version of BaseZImageTransformerBlock.
    
    This is the main transformer block that receives hints from control blocks.
    """
    
    def __init__(
        self, 
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        block_id: Optional[int] = None,  # None means no control hint injection
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.block_id = block_id
        self.modulation = modulation
        
        # Attention
        from diffusers.models.attention_processor import Attention
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(
            query_dim=dim,
            heads=n_heads,
            kv_heads=n_kv_heads,
            dim_head=dim // n_heads,
            bias=False,
            out_bias=True,
            qk_norm="rms_norm" if qk_norm else None,
        )
        
        # FFN
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        
        # Modulation
        if modulation:
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        adaln_input: Optional[torch.Tensor] = None,
        hints: Optional[torch.Tensor] = None,  # [num_hints, B, S, D]
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with optional hint injection.
        
        Args:
            hidden_states: [B, S, D]
            hints: [num_hints, B, S, D] - hints from control blocks
            context_scale: Scale factor for hint injection
        """
        # Apply modulation if enabled
        if self.modulation and adaln_input is not None:
            adaln_out = self.adaln(adaln_input)
            shift1, scale1, shift2, scale2 = adaln_out.chunk(4, dim=-1)
            
            x_normed = self.attention_norm1(hidden_states) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            hidden_states = hidden_states + self.attention(x_normed, attention_mask=attn_mask, freqs_cis=freqs_cis)
            
            x_ffn = self.ffn_norm1(hidden_states) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            hidden_states = hidden_states + self.ffn_norm2(self.feed_forward(x_ffn))
        else:
            hidden_states = hidden_states + self.attention(
                self.attention_norm1(hidden_states), attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            hidden_states = hidden_states + self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(hidden_states))
            )
        
        # Inject hint if this block has a corresponding control block
        if self.block_id is not None and hints is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        
        return hidden_states


class OVZImageControlTransformer2DModel(ZImageTransformer2DModel):
    """
    OV-friendly version of ZImageControlTransformer2DModel.
    
    This model extends the base ZImageTransformer2DModel with control branches
    that can be exported to OpenVINO IR.
    
    Key differences from VideoX-Fun's version:
    1. Uses static tensor operations instead of dynamic list operations
    2. Hints are passed as stacked tensors [num_hints, B, S, D] instead of tuples
    3. Control blocks use OV-compatible forward signatures
    """
    
    @register_to_config
    def __init__(
        self,
        # Control-specific params
        control_layers_places: Optional[List[int]] = None,
        control_refiner_layers_places: Optional[List[int]] = None,
        control_in_dim: Optional[int] = None,
        add_control_noise_refiner: bool = False,
        add_control_noise_refiner_correctly: bool = False,  # VideoX-Fun compat
        # Base model params
        all_patch_size: Tuple[int, ...] = (2,),
        all_f_patch_size: Tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1024, 512, 512],
    ):
        # Initialize base model
        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )
        
        # Control configuration
        self.control_layers_places = control_layers_places or [i for i in range(0, n_layers, 2)]
        self.control_refiner_layers_places = control_refiner_layers_places or list(range(n_refiner_layers))
        self.control_in_dim = control_in_dim or in_channels
        self.add_control_noise_refiner = add_control_noise_refiner
        self.add_control_noise_refiner_correctly = add_control_noise_refiner_correctly
        
        # Mapping from layer index to control block index
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}
        self.control_refiner_layers_mapping = {i: n for n, i in enumerate(self.control_refiner_layers_places)}
        
        # Replace base layers with OV-friendly versions that support hint injection
        del self.layers
        self.layers = nn.ModuleList([
            OVBaseZImageTransformerBlock(
                layer_id=i,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                modulation=True,
                block_id=self.control_layers_mapping.get(i),
            )
            for i in range(n_layers)
        ])
        
        # Control transformer blocks
        self.control_layers = nn.ModuleList([
            OVZImageControlTransformerBlock(
                layer_id=i,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                modulation=True,
                block_id=idx,
            )
            for idx, i in enumerate(self.control_layers_places)
        ])
        
        # Control patch embeddings
        control_x_embedder = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * self.control_in_dim,
                dim,
                bias=True
            )
            control_x_embedder[f"{patch_size}-{f_patch_size}"] = embedder
        self.control_all_x_embedder = nn.ModuleDict(control_x_embedder)
        
        # Control noise refiner (optional)
        if add_control_noise_refiner:
            # Replace noise_refiner with OV-friendly version
            del self.noise_refiner
            self.noise_refiner = nn.ModuleList([
                OVBaseZImageTransformerBlock(
                    layer_id=1000 + i,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    modulation=True,
                    block_id=self.control_refiner_layers_mapping.get(i),
                )
                for i in range(n_refiner_layers)
            ])
            
            self.control_noise_refiner = nn.ModuleList([
                OVZImageControlTransformerBlock(
                    layer_id=1000 + i,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    modulation=True,
                    block_id=i,
                )
                for i in range(n_refiner_layers)
            ])
        else:
            # Use standard blocks for control noise refiner
            self.control_noise_refiner = nn.ModuleList([
                ZImageTransformerBlock(
                    layer_id=1000 + i,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    modulation=True,
                )
                for i in range(n_refiner_layers)
            ])
    
    def load_control_weights(self, safetensors_path: str, strict: bool = False) -> Tuple[List[str], List[str]]:
        """
        Load control weights from a safetensors file.
        
        Args:
            safetensors_path: Path to the .safetensors file
            strict: If True, raise error on missing/unexpected keys
            
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        from safetensors.torch import load_file
        
        state_dict = load_file(safetensors_path)
        result = self.load_state_dict(state_dict, strict=strict)
        
        return result.missing_keys, result.unexpected_keys
    
    @classmethod
    def from_base_and_control(
        cls,
        base_model_path: str,
        control_weights_path: str,
        **kwargs
    ) -> "OVZImageControlTransformer2DModel":
        """
        Create model by loading base weights and then control weights.
        
        Args:
            base_model_path: Path to base ZImageTransformer2DModel
            control_weights_path: Path to control .safetensors
            **kwargs: Override config params
        """
        from diffusers import ZImageTransformer2DModel as BaseModel
        
        # Load base model config
        base_model = BaseModel.from_pretrained(base_model_path)
        base_config = base_model.config
        
        # Create control model with base config + overrides
        config = dict(base_config)
        config.update(kwargs)
        
        model = cls(**config)
        
        # Copy base weights
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        # Load control weights
        model.load_control_weights(control_weights_path, strict=False)
        
        del base_model
        return model
    
    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        control_context: Optional[List[torch.Tensor]] = None,
        control_context_scale: float = 1.0,
    ):
        """
        Forward pass with control context support.
        
        This is a simplified version that calls the base forward and adds control hints.
        For full OV export, use the patched version via ZImageControlTransformerModelPatcher.
        
        Args:
            x: List of latent tensors [C, F, H, W]
            t: Timestep tensor [B]
            cap_feats: List of caption features [S, D]
            patch_size: Patch size for patchification
            f_patch_size: Frame patch size
            control_context: List of control context tensors [C, F, H, W]
            control_context_scale: Scale factor for control hints
            
        Returns:
            Tuple of (output_list, empty_dict)
        """
        # For now, just call the base forward without control
        # The full control logic requires the patcher for OV export
        # This is a placeholder for PyTorch-side testing
        
        # Call parent forward (ignoring control for now)
        return super().forward(x, t, cap_feats, patch_size, f_patch_size)
