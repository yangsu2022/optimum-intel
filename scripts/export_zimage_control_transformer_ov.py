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
        description="Export OV-friendly ZImage Control Transformer to OpenVINO IR."
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help="Base ZImage PT model directory (e.g., Tongyi-MAI/Z-Image-Turbo). "
             "If not set, uses --ov-model-dir config + random backbone weights.",
    )
    parser.add_argument(
        "--ov-model-dir",
        type=Path,
        default=default_ov_model_dir,
        help="OV model root containing transformer/config.json.",
    )
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=default_videox_root
        / "models"
        / "Personalized_Model"
        / "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
        help="Control transformer safetensors path.",
    )
    parser.add_argument(
        "--videox-root",
        type=Path,
        default=default_videox_root,
        help="VideoX-Fun repository root (for config yaml).",
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
        help="Output directory where transformer IR is saved."
    )
    parser.add_argument(
        "--export-type",
        choices=["hints", "full"],
        default="hints",
        help="Export only ControlNet hints subgraph (hints) or full transformer (full).",
    )
    parser.add_argument(
        "--compress-to-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save IR constants in FP16 to reduce model size.",
    )
    parser.add_argument("--height", type=int, default=64, help="Image height for tracing.")
    parser.add_argument("--width", type=int, default=64, help="Image width for tracing.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tracing.")
    parser.add_argument("--text-len", type=int, default=16, help="Text sequence length for tracing.")
    parser.add_argument("--control-context-scale", type=float, default=0.9, help="Control scale for tracing.")
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
    required = [args.safetensors, args.config]
    if args.base_model_dir:
        required.append(args.base_model_dir)
    else:
        required.append(args.ov_model_dir / "transformer" / "config.json")
    
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(missing))


def main():
    args = parse_args()
    _validate_paths(args)

    # Prefer local optimum-intel sources over site-packages.
    script_path = Path(__file__).resolve()
    optimum_intel_root = script_path.parents[1]
    sys.path.insert(0, str(optimum_intel_root))
    # Remove already-loaded site-packages optimum modules so imports resolve to local tree.
    for name in list(sys.modules):
        if name == "optimum" or name.startswith("optimum."):
            del sys.modules[name]

    import gc
    import os
    import openvino as ov
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from optimum.exporters.openvino.model_patcher import ZImageControlTransformerModelPatcher
    from optimum.exporters.openvino.z_image_control_model import OVZImageControlTransformer2DModel

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

    # Load control config from VideoX-Fun
    cfg = OmegaConf.load(str(args.config))
    control_kwargs = OmegaConf.to_container(cfg.get("transformer_additional_kwargs", {}))

    # Build base transformer config
    cfg_path = args.ov_model_dir / "transformer" / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    
    # Remove private keys and merge with control kwargs
    base_cfg = {k: v for k, v in base_cfg.items() if not k.startswith("_")}
    merged_cfg = {**base_cfg, **control_kwargs}
    
    print("Building OVZImageControlTransformer2DModel...")
    transformer = OVZImageControlTransformer2DModel(**merged_cfg)
    
    if torch_dtype != torch.float32:
        transformer = transformer.to(torch_dtype)
    _mem("after build model")

    # Load control weights
    print(f"Loading control weights from {args.safetensors}...")
    state_dict = load_file(str(args.safetensors))
    state_dict = state_dict.get("state_dict", state_dict)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Loaded control weights. missing={len(missing)} unexpected={len(unexpected)}")
    del state_dict
    gc.collect()
    _mem("after load weights")

    # If base model provided, load backbone weights
    if args.base_model_dir:
        print(f"Loading base model weights from {args.base_model_dir}...")
        from diffusers import ZImageTransformer2DModel
        base_model = ZImageTransformer2DModel.from_pretrained(
            str(args.base_model_dir),
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )
        # Copy base weights (won't overwrite control weights)
        base_state = base_model.state_dict()
        transformer.load_state_dict(base_state, strict=False)
        del base_model, base_state
        gc.collect()
        _mem("after load base weights")

    transformer = transformer.eval()

    # Prepare example inputs
    vae_scale_factor = 8
    latent_h = args.height // vae_scale_factor
    latent_w = args.width // vae_scale_factor
    in_channels = transformer.in_channels
    control_in_dim = transformer.control_in_dim  # Use control_in_dim for control_context
    cap_feat_dim = transformer.cap_embedder[1].in_features

    hidden_states = torch.randn(args.batch_size, in_channels, 1, latent_h, latent_w, dtype=torch_dtype)
    timestep = torch.rand(args.batch_size, dtype=torch.float32)
    encoder_hidden_states = torch.randn(args.batch_size, args.text_len, cap_feat_dim, dtype=torch_dtype)
    # control_context uses control_in_dim (33 = 16 + mask + inpaint channels)
    control_context = torch.randn(args.batch_size, control_in_dim, 1, latent_h, latent_w, dtype=torch_dtype)
    control_context_scale = torch.tensor([args.control_context_scale], dtype=torch.float32)

    example_input = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "control_context": control_context,
        "control_context_scale": control_context_scale,
    }

    class _DummyOnnxConfig:
        PATCHING_SPECS = []
        outputs = {"hints": {0: "num_hints"}} if args.export_type == "hints" else {"unified_results": {0: "batch"}}
        torch_to_onnx_output_map = {}
        use_past = False

    transformer._ov_hints_only = args.export_type == "hints"

    print("Converting to OpenVINO...")
    with ZImageControlTransformerModelPatcher(_DummyOnnxConfig(), transformer):
        with torch.no_grad():
            # Use trace with check_trace=False to handle random backbone weights.
            traced = torch.jit.trace(
                transformer,
                example_kwarg_inputs=example_input,
                check_trace=False,
                strict=False,
            )
    _mem("after torch.jit.trace")

    print("Converting traced module to OpenVINO IR...")
    ov_model = ov.convert_model(traced)
    _mem("after ov.convert_model")

    # Save outputs
    output_subdir = "transformer_hints" if args.export_type == "hints" else "transformer"
    output_dir = args.output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer.save_config(str(output_dir))
    
    del traced, transformer
    gc.collect()
    _mem("after free torch module")

    xml_path = output_dir / "openvino_model.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=args.compress_to_fp16)
    _mem("after save_model")

    # Print IR info
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

