import argparse
import os
from typing import Dict, List

from tqdm import tqdm

from leap_llm.apis.calibration.data_loader import load_image_data, load_text_data
from leap_llm.apis.model.gemma4 import _load_gemma4_image_data
from leap_llm.apis.model.siglip import _load_siglip_image_data
from leap_llm.apis.verifier.backends import (
    Backend,
    GEMMA4_MODELS,
    SIGLIP_MODELS,
    TensorDict,
)
from leap_llm.apis.verifier.comparison_reporter import ComparisonReporter
from leap_llm.apis.verifier.types import VerifierArgs
from leap_llm.apis.verifier.utils import time_block

# Hardcoded supported models matching oellm_build
SUPPORTED_MODELS = [
    "deepseek-qwen-1_5b",
    "deepseek-qwen-7b",
    "internvl2-2b",
    "internvl2_5-2b",
    "internvl2-1b",
    "internvl2_5-1b",
    "siglip-so400m",
    "gemma4-e2b-vision",
]


def _validate_args(args: VerifierArgs) -> None:
    """Validate the verifier arguments based on the comparison mode."""
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"Model directory not found: {args.model_dir}")
    if args.compare_mode not in {"bc", "hbm"}:
        raise ValueError("compare_mode must be either 'bc' or 'hbm'")

    # Check if model paths are provided for the corresponding mode
    if args.compare_mode == "bc":
        if not (args.quant_llm_model_path or args.quant_vlm_model_path):
            raise ValueError(
                "BC mode requires at least one of: quant_llm_model_path "
                "or quant_vlm_model_path"
            )
    elif args.compare_mode == "hbm":
        if not (args.hbm_llm_model_path or args.hbm_vlm_model_path or args.remote_ip):
            raise ValueError(
                "HBM mode requires at least one of: hbm_llm_model_path or "
                "hbm_vlm_model_path or remote_ip"
            )


def _get_inference_results(
    compare_mode: str,
    torch_results: tuple[TensorDict, TensorDict] | None,
    comparison_results: tuple[TensorDict, TensorDict | None] | None,
) -> List[TensorDict | None]:
    """
    Structure inference results for the comparison reporter.

    Args:
        compare_mode: Comparison mode ("bc" or "hbm")
        torch_results: Tuple of (output, layers) from torch backend, or None
        comparison_results: Tuple of (output, layers) from comparison backend, or None

    Returns:
        List of tensors ordered for the reporter based on compare_mode
    """
    # Handle missing torch results (should not happen in practice)
    if torch_results is None:
        torch_out, torch_layers = None, None
    else:
        torch_out, torch_layers = torch_results

    # Handle missing comparison results
    if comparison_results is None:
        comp_out, comp_layers = None, None
    else:
        comp_out, comp_layers = comparison_results

    if compare_mode == "bc":
        # BC mode returns both output and layers
        return [torch_out, comp_out, torch_layers, comp_layers]
    elif compare_mode == "hbm":
        # HBM mode returns only output
        return [torch_out, comp_out]
    else:
        raise ValueError(f"Invalid compare mode: {compare_mode}")


def verify_model(verifier_args: VerifierArgs):
    """Verify model outputs by running inference and generating reports."""
    _validate_args(verifier_args)

    # Initialize unified backend
    backend = Backend(verifier_args)
    reporter = ComparisonReporter(verifier_args.model_name)

    # Determine which models to run based on provided paths
    if verifier_args.compare_mode == "bc":
        # BC mode: check BC-related paths
        run_llm = bool(verifier_args.quant_llm_model_path)
        run_vlm = bool(verifier_args.quant_vlm_model_path)
    elif verifier_args.compare_mode == "hbm":
        # HBM mode: check HBM-related paths
        run_llm = bool(verifier_args.hbm_llm_model_path)
        run_vlm = bool(verifier_args.hbm_vlm_model_path)
    else:
        run_llm = run_vlm = False

    if run_llm:
        if backend.torch_llm_model is None:
            path_kind = (
                "quant_model_path"
                if verifier_args.compare_mode == "bc"
                else "hbm_model_path"
            )
            raise ValueError(
                f"Model {verifier_args.model_name} does not support LLM, "
                f"but LLM comparison was requested via {path_kind}"
            )

        for i, text_prompt in enumerate(
            tqdm(
                load_text_data(verifier_args.input_text_path),
                desc="Processing text prompts",
            )
        ):
            with time_block(f"LLM total prompt_{i}"):
                last_valid_step = backend.compute_last_valid_step_index(text_prompt)
                torch_results = backend.run_llm_torch(text_prompt)
                comparison_results = backend.run_llm(text_prompt)

                llm_inference_results: Dict[str, List[TensorDict | None]] = {
                    "llm": _get_inference_results(
                        verifier_args.compare_mode, torch_results, comparison_results
                    )
                }
                reporter.compare_inference_results(
                    llm_inference_results,
                    verifier_args.compare_mode,
                    prompt_id=f"prompt_{i}",
                    last_valid_step=last_valid_step,
                )

    if run_vlm:
        if backend.torch_vlm_model is None:
            path_kind = (
                "quant_vlm_model_path"
                if verifier_args.compare_mode == "bc"
                else "hbm_vlm_model_path"
            )
            raise ValueError(
                f"Model {verifier_args.model_name} does not support VLM, "
                f"but VLM comparison was requested via {path_kind}"
            )

        if verifier_args.model_name in GEMMA4_MODELS:
            image_loader = _load_gemma4_image_data(verifier_args.input_image_path)
        elif verifier_args.model_name in SIGLIP_MODELS:
            image_loader = _load_siglip_image_data(verifier_args.input_image_path)
        else:
            image_loader = load_image_data(verifier_args.input_image_path, max_num=1)

        for i, image_tensor in enumerate(
            tqdm(
                image_loader,
                desc="Processing images",
            )
        ):
            with time_block(f"VLM total image_{i}"):
                torch_results = backend.run_vlm_torch(image_tensor)
                comparison_results = backend.run_vlm(image_tensor)

                vlm_inference_results: Dict[str, List[TensorDict | None]] = {
                    "vlm": _get_inference_results(
                        verifier_args.compare_mode, torch_results, comparison_results
                    )
                }
                reporter.compare_inference_results(
                    vlm_inference_results,
                    verifier_args.compare_mode,
                    image_id=f"image_{i}",
                )

    reporter.generate_reports()


def main():
    """Parse command-line arguments and run the verifier."""
    parser = argparse.ArgumentParser(
        description="Verify model accuracy by comparing PyTorch and Quantized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (required)",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Original model directory (required)",
    )
    parser.add_argument(
        "--quant_llm_model_path",
        type=str,
        required=False,
        help="Path to the quantized LLM model for BC comparison (required for BC mode)",
    )
    parser.add_argument(
        "--quant_vlm_model_path",
        type=str,
        required=False,
        help="Path to the quantized VLM model for BC comparison (required for BC mode)",
    )
    parser.add_argument(
        "--hbm_llm_model_path",
        type=str,
        required=False,
        help="Path to HBM model file for LLM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--hbm_vlm_model_path",
        type=str,
        required=False,
        help="Path to HBM model file for VLM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--input_text_path",
        type=str,
        required=False,
        help="Path to text data for input (Optional)",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        required=False,
        help="Path to image data for input (Optional)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=256,
        help="Chunk size, default is 256 (optional)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        required=False,
        default=4096,
        help="Cache length, default is 4096 (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device to run on, default is cpu (optional)",
    )
    parser.add_argument(
        "--remote_ip",
        type=str,
        required=False,
        help="Remote IP address for HBM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=False,
        default="root",
        help=(
            "Username for remote HBM connection.(Required for HBM mode, "
            "default is root)"
        ),
    )
    parser.add_argument(
        "--password",
        type=str,
        required=False,
        default="",
        help=(
            "Password for remote HBM connection.(Required for HBM mode, "
            "default is empty)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=22,
        help="Port for remote HBM connection.(Required for HBM mode, default is 22)",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        required=False,
        default="/tmp/",
        help="Remote path for HBM inference.(Required for HBM mode, default is /tmp/)",
    )
    args = parser.parse_args()

    # Automatically determine compare_mode based on provided paths
    compare_mode = None
    if args.quant_llm_model_path or args.quant_vlm_model_path:
        compare_mode = "bc"
    elif args.hbm_llm_model_path or args.hbm_vlm_model_path:
        compare_mode = "hbm"
    else:
        raise ValueError(
            "No comparison model path provided. Please specify at least one of: "
            "quant_llm_model_path, quant_vlm_model_path, hbm_llm_model_path, "
            "or hbm_vlm_model_path"
        )

    verifier_args = VerifierArgs(
        model_name=args.model_name,
        model_dir=args.model_dir,
        compare_mode=compare_mode,
        input_text_path=args.input_text_path,
        input_image_path=args.input_image_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        quant_llm_model_path=args.quant_llm_model_path,
        quant_vlm_model_path=args.quant_vlm_model_path,
        hbm_llm_model_path=args.hbm_llm_model_path,
        hbm_vlm_model_path=args.hbm_vlm_model_path,
        remote_ip=args.remote_ip,
        username=args.username,
        password=args.password,
        port=args.port,
        remote_path=args.remote_path,
    )

    try:
        verify_model(verifier_args)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
