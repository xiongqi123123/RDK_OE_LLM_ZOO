import argparse
import os
import sys
from pathlib import Path

os.environ["DEV_B30_TRITON_VPU"] = "1"
os.environ["DEV_B30_ENABLE_VPU_EXTRA_OP"] = "1"
os.environ["DEV_B30_ENABLE_VPU_TRIAL_OP"] = "1"

sys.path.append("../../")
from leap_llm.apis.model.model_factory import (  # noqa: E402
    create_model_api,
    get_marches_with_model,
    get_supported_marches,
    get_supported_models,
)

DEFAULT_COMPILE_KWARGS = {
    "march": "nash-m",
    "jobs": 32,
    "progress_bar": True,
    "max_time_per_fc": 0.0,
    "opt": 2,
    "debug": False,
    "advice": 0.0,
    "balance": 100,
    "input_no_padding": False,
    "output_no_padding": False,
    "cache_mode": "disable",
    "cache_path": "",
}


def validated_path(check_exists=True):
    def validator(path_string):
        if not path_string:
            raise argparse.ArgumentTypeError("Path cannot be empty")

        path = Path(os.path.expanduser(os.path.expandvars(path_string)))

        if check_exists and not path.exists():
            raise argparse.ArgumentTypeError(f"Path does not exist: {path}")

        return str(path.resolve())

    return validator


def main():
    # set model_name help string
    model_help_parts = [
        f"    - {model}: {', '.join(get_marches_with_model(model))}"
        for model in get_supported_models()
    ]
    model_help = "Model name. Supported models and their marches:\n" + "\n".join(
        model_help_parts
    )

    parser = argparse.ArgumentParser(
        description="Compile a Large Language Model for deployment on hardware.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=model_help,
    )
    parser.add_argument(
        "--march",
        type=str,
        required=True,
        choices=get_supported_marches(),
        help="Target hardware architecture for compilation. (Required)",
    )
    parser.add_argument(
        "--input_model_path",
        type=validated_path(check_exists=True),
        required=True,
        help="Path to the source model directory. (Required)",
    )
    parser.add_argument(
        "--output_model_path",
        type=validated_path(check_exists=False),
        required=True,
        help="Path to save the compiled model. (Required)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        default=4096,
        help="Maximum sequence length for the KV-cache. (default: 4096). "
        "Note: cache_len must be an integer multiple of chunk_size.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Number of tokens per prefill chunk. (default: 256)",
    )
    parser.add_argument(
        "--max_time_per_fc",
        type=float,
        default=0.0,
        help="Set maximum time constraint (unit:us) for per funccall. (default: 0.0). ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Compute device: 'cpu' (default), 'cuda'/'cuda:0', or 'cuda:<index>'. "
            "Using CUDA on x86 can accelerate calibration."
        ),
    )
    parser.add_argument(
        "--calib_image_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration dataset for vision models. (Optional)",
    )
    parser.add_argument(
        "--calib_text_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration JSON file or directory of JSON files. (Optional)",
    )
    parser.add_argument(
        "--calib_conversation_path",
        type=validated_path(check_exists=True),
        default=None,
        help=("Path to the conversation for Qwen2.5-Omni. (Optional)"),
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=8,
        help=(
            "Weight quantization bits, 4 or 8. Applies only to InternVL models "
            "(e.g., internvl2-2b, internvl2_5-2b); ignored for other models. (Optional)"
        ),
    )
    parser.add_argument(
        "--weight_scales_file",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the weight scales file. (Optional)",
    )
    parser.add_argument(
        "-v",
        "--verifier",
        action="store_true",
        help=(
            "Run consistency verification after compilation using the built-in "
            "verifier tool. (Optional)"
        ),
    )
    parser.add_argument(
        "--remote_ip",
        type=str,
        default=None,
        help="Remote IP address for HBM model. (Optional)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22,
        help="Port for remote HBM connection",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="root",
        help="Username for remote HBM connection",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Password for remote HBM connection",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default="/userdata/data/hbm_infer",
        help="Remote path for HBM inference",
    )
    parser.add_argument(
        "--vit_core_num",
        type=int,
        default=1,
        help="VIT model multi bpu core num, only for j6p & vlm model",
    )
    parser.add_argument(
        "--prefill_core_num",
        type=int,
        default=1,
        help="LLM prefill model multi bpu core num, only for j6p",
    )
    parser.add_argument(
        "--decode_core_num",
        type=int,
        default=1,
        help="LLM decode model multi bpu core num, only for j6p",
    )

    args = parser.parse_args()

    if not 256 <= args.cache_len <= 4096:
        parser.error(
            f"--cache_len ({args.cache_len}) must be within the range [256, 4096]!"
        )

    if not 128 <= args.chunk_size <= 2048:
        parser.error(
            f"--chunk_size ({args.chunk_size}) must be within the range [128, 2048]!"
        )

    if args.cache_len <= args.chunk_size or args.cache_len % args.chunk_size != 0:
        parser.error(
            f"--cache_len ({args.cache_len}) must be greater than "
            f"--chunk_size ({args.chunk_size}) and must be a multiple of "
            f"--chunk_size ({args.chunk_size})!"
        )

    if args.w_bits != 4 and args.w_bits != 8 and "internvl" in args.model_name:
        parser.error(f"--w_bits ({args.w_bits}) must be 4 or 8.")

    # if need add parser args in compile, only add parser
    compile_kwargs = DEFAULT_COMPILE_KWARGS.copy()
    for k, v in args.__dict__.items():
        if k not in compile_kwargs:
            continue
        compile_kwargs[k] = v
    if args.model_name == "qwen2_5-omni-3b":
        compile_kwargs["input_no_padding"] = True
        compile_kwargs["output_no_padding"] = True

    if "internvl" in args.model_name:
        valid_values = [1, 2, 4]

        if args.vit_core_num not in valid_values:
            parser.error("vit_core_num must be one of [1, 2, 4] for vlm models")
        if args.prefill_core_num not in valid_values:
            parser.error("prefill_core_num must be one of [1, 2, 4] for internvl")
        if args.decode_core_num not in valid_values:
            parser.error("decode_core_num must be one of [1, 2, 4] for internvl")

    model = create_model_api(args.model_name, args)
    if not model:
        return
    model.compile(**compile_kwargs)

    if not args.verifier:
        return

    from leap_llm.apis.verifier.types import VerifierArgs
    from leap_llm.apis.verifier_cli import verify_model

    # Determine quantized BC file paths from API helper when available.
    hbm_llm_model_path: str | None = None
    hbm_vlm_model_path: str | None = None

    if hasattr(model, "get_hbm_path"):
        paths = model.get_hbm_path()
        if len(paths) == 1:
            # Vision-only models (e.g., SigLIP) have a single VLM path
            if "siglip" in args.model_name or "gemma4" in args.model_name:
                hbm_vlm_model_path = paths[0]
            else:
                hbm_llm_model_path = paths[0]
        elif len(paths) == 2:
            hbm_llm_model_path, hbm_vlm_model_path = paths

    verifier_args = VerifierArgs(
        model_name=args.model_name,
        model_dir=args.input_model_path,
        compare_mode="hbm",
        input_text_path=args.calib_text_path,
        input_image_path=args.calib_image_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        hbm_llm_model_path=hbm_llm_model_path,
        hbm_vlm_model_path=hbm_vlm_model_path,
        remote_ip=args.remote_ip,
        port=args.port,
        username=args.username,
        password=args.password,
        remote_path=args.remote_path,
    )

    verify_model(verifier_args)


if __name__ == "__main__":
    main()
