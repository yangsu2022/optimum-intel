"""Export VideoX-Fun Z-Image control transformer (hints subgraph) to OpenVINO IR.

Uses ZImageControlTransformerModelPatcher for OV-friendly graph structure,
TorchScriptPythonDecoder for proper conversion, and fixed partial shapes
for GPU f16 compatibility.
"""
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

import argparse
import gc
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VideoX-Fun Z-Image control transformer to OpenVINO IR.")
    parser.add_argument("--model-name", type=Path, required=True, help="Base Z-Image PT model directory (e.g. Z-Image-Turbo-hf).")
    parser.add_argument("--safetensors", type=Path, required=True, help="ControlNet weights safetensors path.")
    parser.add_argument("--videox-root", type=Path, required=True, help="VideoX-Fun repo root (contains videox_fun/ and config/).")
    parser.add_argument("--config", type=Path, default=None, help="Model config yaml. Default: <videox-root>/config/z_image/z_image_control_2.1_lite.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512, help="Image height for tracing (must match target).")
    parser.add_argument("--width", type=int, default=512, help="Image width for tracing (must match target).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-len", type=int, default=128, help="Cap sequence length (ZIMAGE_CAP_SEQ=128).")
    parser.add_argument("--control-context-scale", type=float, default=0.9)
    parser.add_argument("--weight-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--enable-int4-compression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--int4-mode", default="int4_asym")
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--int4-ratio", type=float, default=1.0)
    parser.add_argument("--compress-to-fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    optimum_intel_root = script_path.parents[1]
    sys.path.insert(0, str(optimum_intel_root))
    for name in list(sys.modules):
        if name == "optimum" or name.startswith("optimum."):
            del sys.modules[name]

    if args.config is None:
        args.config = args.videox_root / "config" / "z_image" / "z_image_control_2.1_lite.yaml"

    project_roots = [args.videox_root, args.videox_root.parent]
    for root in project_roots:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    import openvino as ov
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    from optimum.exporters.openvino.model_patcher import ZImageControlTransformerModelPatcher

    torch_dtype = _resolve_torch_dtype(args.weight_dtype)

    cfg = OmegaConf.load(str(args.config))
    control_kwargs = OmegaConf.to_container(cfg["transformer_additional_kwargs"])

    # Build control transformer
    from videox_fun.models import ZImageControlTransformer2DModel

    print(f"Loading from pretrained: {args.model_name}")
    transformer = ZImageControlTransformer2DModel.from_pretrained(
        str(args.model_name), subfolder="transformer",
        low_cpu_mem_usage=True, torch_dtype=torch_dtype,
        transformer_additional_kwargs=control_kwargs,
    )

    # Load control weights
    state_dict = load_file(str(args.safetensors))
    state_dict = state_dict.get("state_dict", state_dict)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Loaded control weights. missing={len(missing)} unexpected={len(unexpected)}")

    transformer = transformer.to(torch_dtype).eval()

    # Set hints-only mode
    transformer._ov_hints_only = True

    vae_scale_factor = 8
    latent_h = args.height // vae_scale_factor
    latent_w = args.width // vae_scale_factor
    cap_feat_dim = int(transformer.cap_embedder[1].in_features)
    in_channels = int(transformer.in_channels)
    control_in_dim = int(transformer.control_in_dim)

    hidden_states = torch.randn(args.batch_size, in_channels, 1, latent_h, latent_w, dtype=torch_dtype)
    timestep = torch.rand(args.batch_size, dtype=torch.float32)
    encoder_hidden_states = torch.randn(args.batch_size, args.text_len, cap_feat_dim, dtype=torch_dtype)
    control_context = torch.randn(args.batch_size, control_in_dim, 1, latent_h, latent_w, dtype=torch_dtype)
    control_context_scale = torch.tensor([args.control_context_scale], dtype=torch.float32)

    example_input = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "control_context": control_context,
        "control_context_scale": control_context_scale,
    }

    input_info = [
        (list(hidden_states.shape), ov.Type.f32),
        (list(timestep.shape), ov.Type.f32),
        (list(encoder_hidden_states.shape), ov.Type.f32),
        (list(control_context.shape), ov.Type.f32),
        (list(control_context_scale.shape), ov.Type.f32),
    ]
    input_names = ["hidden_states", "timestep", "encoder_hidden_states", "control_context", "control_context_scale"]

    class _DummyOnnxConfig:
        PATCHING_SPECS = []
        outputs = {"hints": {0: "num_hints"}}
        torch_to_onnx_output_map = {}
        use_past = False

    print("Applying ZImageControlTransformerModelPatcher and converting...")
    with ZImageControlTransformerModelPatcher(_DummyOnnxConfig(), transformer):
        with torch.no_grad():
            ts_decoder = TorchScriptPythonDecoder(
                transformer,
                example_input=example_input,
                trace_kwargs={"check_trace": False},
            )
            ov_model = ov.convert_model(
                ts_decoder,
                example_input=example_input,
                input=input_info,
            )
    print("Conversion done.")

    # Set input names
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    # Fix partial shapes for GPU f16 compatibility
    partial_shapes = {
        "hidden_states": ov.PartialShape([-1, in_channels, -1, -1, -1]),
        "encoder_hidden_states": ov.PartialShape([-1, -1, cap_feat_dim]),
        "control_context": ov.PartialShape([-1, control_in_dim, -1, -1, -1]),
        "control_context_scale": ov.PartialShape([-1]),
        "timestep": ov.PartialShape([-1]),
    }
    reshape_map = {}
    for inp in ov_model.inputs:
        name = inp.get_any_name()
        if name in partial_shapes:
            reshape_map[inp] = partial_shapes[name]
    if reshape_map:
        print("Reshaping for GPU f16:")
        for inp, ps in reshape_map.items():
            print(f"  {inp.get_any_name()}: {inp.get_partial_shape()} -> {ps}")
        ov_model.reshape(reshape_map)

    # Save
    output_dir = args.output_dir / "transformer_hints"
    output_dir.mkdir(parents=True, exist_ok=True)

    del ts_decoder
    gc.collect()

    if args.enable_int4_compression:
        import nncf
        print(f"Applying NNCF int4 (mode={args.int4_mode}, gs={args.int4_group_size}, ratio={args.int4_ratio})...")
        ov_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode(args.int4_mode),
            group_size=args.int4_group_size,
            ratio=args.int4_ratio,
            advanced_parameters=nncf.AdvancedCompressionParameters(
                group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST
            ),
        )

    xml_path = output_dir / "openvino_model.xml"
    save_fp16 = args.compress_to_fp16 and not args.enable_int4_compression
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=save_fp16)

    print(f"\nSaved: {xml_path}")
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
