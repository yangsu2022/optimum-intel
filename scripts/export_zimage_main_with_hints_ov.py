import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[3]
    default_videox_root = workspace_root / "VideoX-Fun"
    default_ov_model_dir = workspace_root / "Z-Image-Turbo-ov-int4-gs64-control"

    parser = argparse.ArgumentParser(
        description="Export Z-Image main transformer IR with explicit hints input."
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        required=True,
        help="Base HF/PT model directory (must contain transformer weights).",
    )
    parser.add_argument(
        "--ov-model-dir",
        type=Path,
        default=default_ov_model_dir,
        help="OV model root containing transformer/config.json.",
    )
    parser.add_argument(
        "--videox-root",
        type=Path,
        default=default_videox_root,
        help="VideoX-Fun repository root.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_videox_root / "config" / "z_image" / "z_image_control_2.1_lite.yaml",
        help="VideoX-Fun config yaml for transformer_additional_kwargs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory where main transformer (with hints input) is saved.",
    )
    parser.add_argument(
        "--compress-to-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save IR constants in FP16.",
    )
    parser.add_argument("--height", type=int, default=64, help="Image height for tracing.")
    parser.add_argument("--width", type=int, default=64, help="Image width for tracing.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tracing.")
    parser.add_argument("--text-len", type=int, default=16, help="Text sequence length for tracing.")
    parser.add_argument(
        "--control-context-scale",
        type=float,
        default=0.9,
        help="Control scale for hint injection in tracing.",
    )
    parser.add_argument(
        "--weight-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Torch dtype for model weights.",
    )
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]


def _validate_paths(args: argparse.Namespace):
    required = [
        args.base_model_dir,
        args.config,
        args.ov_model_dir / "transformer" / "config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(missing))


def main():
    args = parse_args()
    _validate_paths(args)

    script_path = Path(__file__).resolve()
    optimum_intel_root = script_path.parents[1]
    sys.path.insert(0, str(optimum_intel_root))
    for name in list(sys.modules):
        if name == "optimum" or name.startswith("optimum."):
            del sys.modules[name]

    import gc
    import os

    import openvino as ov
    import torch
    from omegaconf import OmegaConf

    from optimum.exporters.openvino.model_patcher import ZImageControlTransformerModelPatcher
    from diffusers import ZImageTransformer2DModel

    try:
        import psutil

        _proc = psutil.Process(os.getpid())

        def _mem(tag: str) -> None:
            print(f"[mem] {tag}: RSS={_proc.memory_info().rss / 1024 ** 3:.2f} GB")

    except ImportError:

        def _mem(tag: str) -> None:
            pass

    _mem("startup")

    torch_dtype = _resolve_torch_dtype(args.weight_dtype)

    cfg = OmegaConf.load(str(args.config))
    control_kwargs = OmegaConf.to_container(cfg.get("transformer_additional_kwargs", {}))

    print(f"Loading base model weights from {args.base_model_dir}...")
    model = ZImageTransformer2DModel.from_pretrained(
        str(args.base_model_dir),
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    gc.collect()
    _mem("after load base model")

    model = model.eval()
    control_layers_places = control_kwargs.get("control_layers_places") or [0, 10, 20]
    control_layers_mapping = {layer_idx: hint_idx for hint_idx, layer_idx in enumerate(control_layers_places)}
    print(f"Using control_layers_places={control_layers_places}")

    def _pad_and_stack(seqs, padding_value=0.0):
        # Avoid aten::pad_sequence in traced graph; OpenVINO PyTorch FE may reject it.
        max_len = max(s.shape[0] for s in seqs)
        out = []
        for s in seqs:
            pad_len = max_len - s.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + tuple(s.shape[1:])
                pad = torch.full(pad_shape, padding_value, dtype=s.dtype, device=s.device)
                s = torch.cat([s, pad], dim=0)
            out.append(s)
        return torch.stack(out, dim=0)

    class MainWithHintsWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            hints: torch.Tensor,
            control_context_scale: torch.Tensor,
        ) -> torch.Tensor:
            x = list(torch.unbind(hidden_states, dim=0))
            t = timestep
            cap_feats = list(torch.unbind(encoder_hidden_states, dim=0))

            patch_size = 2
            f_patch_size = 1

            bsz = len(x)
            device = x[0].device

            t = t * self.m.t_scale
            t = self.m.t_embedder(t)

            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_inner_pad_mask,
                cap_inner_pad_mask,
            ) = self.m.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

            x_item_seqlens = [_.shape[0] for _ in x]
            x_max_item_seqlen = max(x_item_seqlens)

            x = torch.cat(x, dim=0)
            x = self.m.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

            adaln_input = t.type_as(x)
            x_flat_mask = torch.cat(x_inner_pad_mask)
            x = torch.where(x_flat_mask.unsqueeze(-1), self.m.x_pad_token, x)
            x = list(x.split(x_item_seqlens, dim=0))
            x_freqs_cis = list(self.m.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

            x = _pad_and_stack(x, padding_value=0.0)
            x_freqs_cis = _pad_and_stack(x_freqs_cis, padding_value=0.0)
            x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(x_item_seqlens):
                x_attn_mask[i, :seq_len] = 1

            for layer in self.m.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

            cap_item_seqlens = [_.shape[0] for _ in cap_feats]
            cap_max_item_seqlen = max(cap_item_seqlens)

            cap_feats = torch.cat(cap_feats, dim=0)
            cap_feats = self.m.cap_embedder(cap_feats)
            cap_flat_mask = torch.cat(cap_inner_pad_mask)
            cap_feats = torch.where(cap_flat_mask.unsqueeze(-1), self.m.cap_pad_token, cap_feats)
            cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
            cap_freqs_cis = list(self.m.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

            cap_feats = _pad_and_stack(cap_feats, padding_value=0.0)
            cap_freqs_cis = _pad_and_stack(cap_freqs_cis, padding_value=0.0)
            cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(cap_item_seqlens):
                cap_attn_mask[i, :seq_len] = 1

            for layer in self.m.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

            unified = []
            unified_freqs_cis = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                cap_len = cap_item_seqlens[i]
                unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
                unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))

            unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
            unified_max_item_seqlen = max(unified_item_seqlens)

            unified = _pad_and_stack(unified, padding_value=0.0)
            unified_freqs_cis = _pad_and_stack(unified_freqs_cis, padding_value=0.0)
            unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_item_seqlens):
                unified_attn_mask[i, :seq_len] = 1

            ctx_scale = control_context_scale.reshape(())

            for layer_idx, layer in enumerate(self.m.layers):
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
                if layer_idx in control_layers_mapping:
                    unified = unified + hints[control_layers_mapping[layer_idx]] * ctx_scale

            unified = self.m.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
            unified = list(unified.unbind(dim=0))
            out = self.m.unpatchify(unified, x_size, patch_size, f_patch_size)
            return out[0]

    wrapper = MainWithHintsWrapper(model).eval()

    vae_scale_factor = 8
    latent_h = args.height // vae_scale_factor
    latent_w = args.width // vae_scale_factor
    in_channels = model.in_channels
    cap_feat_dim = model.cap_embedder[1].in_features
    num_hints = len(control_layers_places)
    dim = model.config.dim

    hidden_states = torch.randn(args.batch_size, in_channels, 1, latent_h, latent_w, dtype=torch_dtype)
    timestep = torch.rand(args.batch_size, dtype=torch.float32)
    encoder_hidden_states = torch.randn(args.batch_size, args.text_len, cap_feat_dim, dtype=torch_dtype)
    # Use seq=1 for tracing; runtime can feed full hints tensor from transformer_hints IR.
    hints = torch.randn(num_hints, args.batch_size, 1, dim, dtype=torch_dtype)
    control_context_scale = torch.tensor([args.control_context_scale], dtype=torch.float32)

    example_input = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "hints": hints,
        "control_context_scale": control_context_scale,
    }

    class _DummyOnnxConfig:
        PATCHING_SPECS = []
        outputs = {"unified_results": {0: "batch"}}
        torch_to_onnx_output_map = {}
        use_past = False

    print("Tracing main-with-hints wrapper...")
    with ZImageControlTransformerModelPatcher(_DummyOnnxConfig(), model):
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                example_kwarg_inputs=example_input,
                check_trace=False,
                strict=False,
            )
    _mem("after torch.jit.trace")

    print("Converting traced module to OpenVINO IR...")
    try:
        ov_model = ov.convert_model(traced)
    except Exception as e:
        print("[convert-error] Failed to convert traced module.")
        print(f"[convert-error] type={type(e).__name__}")
        print(f"[convert-error] message={e}")
        raise
    _mem("after ov.convert_model")

    output_dir = args.output_dir / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_config(str(output_dir))

    del traced, wrapper, model
    gc.collect()
    _mem("after free torch module")

    xml_path = output_dir / "openvino_model.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=args.compress_to_fp16)
    _mem("after save_model")

    print(f"\nSaved transformer IR: {xml_path}")
    print("Inputs:")
    for inp in ov_model.inputs:
        print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
    print("Outputs:")
    for idx, out in enumerate(ov_model.outputs):
        try:
            name = out.get_any_name()
        except RuntimeError:
            name = f"output_{idx}"
        print(f"  {name}: {out.get_partial_shape()}")


if __name__ == "__main__":
    main()
