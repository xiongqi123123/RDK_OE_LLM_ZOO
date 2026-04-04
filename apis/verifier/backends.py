from __future__ import annotations

import os
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from hbdk4.compiler import Hbm, load
from hbdk4.compiler.hbm import Graph
from hbdk4.compiler.overlay import Module, Value
from transformers import AutoModelForCausalLM, AutoTokenizer

from leap_llm.apis.calibration import CalibrationDataPreparer
from leap_llm.apis.verifier.types import TensorDict, TensorInfo, VerifierArgs
from leap_llm.apis.verifier.utils import cast_to_tensor_info, time_block
from leap_llm.models.deepseek.model import DeepSeek
from leap_llm.models.internvl_1b.model import Internlm1b, Internvl1bVision
from leap_llm.models.internvl_2b.model import Internlm2b, Internvl2bVision
from leap_llm.models.siglip.model import SiglipVision

DEEPSEEK_MODELS = ["deepseek-qwen-1_5b", "deepseek-qwen-7b"]
INTERNVL_MODELS = ["internvl2-1b", "internvl2-2b", "internvl2_5-1b", "internvl2_5-2b"]
SIGLIP_MODELS = ["siglip-so400m"]
padding_side_dict = {
    "internvl2-1b": "right",
    "internvl2-2b": "right",
    "internvl2_5-1b": "right",
    "internvl2_5-2b": "right",
    "deepseek-qwen-1_5b": "left",
    "deepseek-qwen-7b": "left",
}

mask_value_dict = {
    "internvl2-1b": -8192,
    "internvl2-2b": -8192,
    "internvl2_5-1b": -8192,
    "internvl2_5-2b": -8192,
    "deepseek-qwen-1_5b": -512,
    "deepseek-qwen-7b": -512,
}


internvl_model_class_dict = {
    "llm": {
        "internvl2-1b": Internlm1b,
        "internvl2-2b": Internlm2b,
        "internvl2_5-1b": Internlm1b,
        "internvl2_5-2b": Internlm2b,
    },
    "vlm": {
        "internvl2-1b": Internvl1bVision,
        "internvl2-2b": Internvl2bVision,
        "internvl2_5-1b": Internvl1bVision,
        "internvl2_5-2b": Internvl2bVision,
    },
}


class Backend:
    """Unified backend that handles inference for Torch, BC, and HBM models."""

    def __init__(self, args: VerifierArgs):
        self.args = args
        self.device = self.args.device

        # Torch models
        self.torch_llm_model = None
        self.torch_vlm_model = None
        self.torch_llm_model_core = None
        self.torch_vlm_model_core = None
        self.tokenizer = None
        self.calib_data_preparer: CalibrationDataPreparer | None = None
        self._prepared_inputs_cache: dict[str, tuple] = {}
        self.torch_layers_outputs: TensorDict = OrderedDict()
        self.torch_vlm_layers_outputs: TensorDict = OrderedDict()
        self.num_hidden_layers = None

        # BC models
        self.bc_model: Module | None = None
        self.bc_vlm_model: Module | None = None
        self.bc_layers_outputs: TensorDict = OrderedDict()
        self.bc_vlm_layers_outputs: TensorDict = OrderedDict()

        # HBM models
        self._hbm_llm_module = None
        self._hbm_vlm_module = None

        # Load all models
        self._load_torch_model()
        self.bc_model = self._load_bc_model(self.args.quant_llm_model_path)
        self.bc_vlm_model = self._load_bc_model(self.args.quant_vlm_model_path)
        self._hbm_llm_module = self._load_hbm_module(self.args.hbm_llm_model_path)
        self._hbm_vlm_module = self._load_hbm_module(self.args.hbm_vlm_model_path)

        # set calib data preparer (not needed for vision-only models)
        if self.args.model_name not in SIGLIP_MODELS:
            self.calib_data_preparer = CalibrationDataPreparer(
                self.args.model_dir,
                seq_len=self.args.chunk_size,
                kv_cache_len=self.args.cache_len,
                device=self.device,
                transpose_cache=self.args.transpose_cache,
                mask_value=mask_value_dict[self.args.model_name],
            )

        # Comparison index bookkeeping
        self.last_compare_index = None  # index within last chunk (0-based)
        self.last_compare_abs_position = None  # absolute token count (1-based)
        self.last_compare_chunk_index_1based = None  # index within chunk (1-based)
        self.last_compare_token_index_1based = None  # absolute (1-based)

    def _load_torch_model(self):
        """Load the torch model."""
        if not os.path.isdir(self.args.model_dir):
            raise ValueError(f"Model directory not found: {self.args.model_dir}")

        ckpt_path = os.path.join(self.args.model_dir, "model_checkpoint.pth")

        # Vision-only models (SigLIP) don't use AutoModelForCausalLM
        if self.args.model_name not in SIGLIP_MODELS:
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_dir, trust_remote_code=True
            )
            torch.save(model.state_dict(), ckpt_path)

        try:
            if self.args.model_name in DEEPSEEK_MODELS:
                self.torch_llm_model = DeepSeek.build(
                    model_dir=self.args.model_dir,
                    chunk_size=self.args.chunk_size,
                    cache_len=self.args.cache_len,
                    preserve_precision=False,
                )
                self.torch_llm_model.model.compile_mode(False)
                self.torch_llm_model.model.to(self.device)
            elif self.args.model_name in INTERNVL_MODELS:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.torch_llm_model = internvl_model_class_dict["llm"][
                    self.args.model_name
                ].load_model(
                    input_model_path=self.args.model_dir,
                    checkpoint=checkpoint,
                    chunk_size=self.args.chunk_size,
                    cache_len=self.args.cache_len,
                )
                self.torch_vlm_model = internvl_model_class_dict["vlm"][
                    self.args.model_name
                ].load_model(
                    input_model_path=self.args.model_dir,
                    checkpoint=checkpoint,
                )
                self.torch_llm_model.model.compile_mode(False)
                self.torch_llm_model.model.to(self.device)
                self.torch_vlm_model.model.compile_mode(False)
                self.torch_vlm_model.model.to(self.device)

            elif self.args.model_name in SIGLIP_MODELS:
                from safetensors import safe_open

                siglip_ckpt = {}
                sf_path = os.path.join(self.args.model_dir, "model.safetensors")
                if os.path.exists(sf_path):
                    with safe_open(sf_path, framework="pt") as f:
                        for key in f.keys():
                            siglip_ckpt[key] = f.get_tensor(key)
                else:
                    siglip_ckpt = torch.load(ckpt_path, map_location=self.device)
                self.torch_vlm_model = SiglipVision.load_model(
                    self.args.model_dir, siglip_ckpt
                )
                self.torch_vlm_model.set_compile_mode(False)
                self.torch_vlm_model.set_model_device(self.device, torch.float32)

            if self.args.model_name not in SIGLIP_MODELS:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_dir, trust_remote_code=True
                )

            self.torch_llm_model_core = self._get_model_core(self.torch_llm_model)
            self.torch_vlm_model_core = self._get_model_core(self.torch_vlm_model)

            # TODO: delete after debug
            # if self.tokenizer and self.tokenizer.pad_token is None:
            #     self.tokenizer.pad_token = (
            #         self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"
            #     )
        except Exception as e:
            raise Exception(f"Failed to load {self.args.model_name} model: {e}")

    def _load_bc_model(self, model_path: str) -> Module | None:
        """Load BC model from path."""
        if not model_path:
            return None

        return load(model_path)

    def _load_hbm_module(self, path: str) -> Graph | None:
        """Load HBM module from path."""
        if not path:
            return None

        return Hbm(path)[0]

    def get_prepared_inputs(self, text_input: str):
        """Return cached prepared input chunks or prepare and cache them."""
        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized.")

        if text_input not in self._prepared_inputs_cache:
            prepared = self.calib_data_preparer.prepare_inputs(text_input)
            self._prepared_inputs_cache[text_input] = prepared
        return self._prepared_inputs_cache[text_input]

    def compute_last_valid_step_index(self, text_input: str) -> int | None:
        """Return the compare index within the last chunk (0-based), minimal cost.

        - chunk_size - 1 if left padding.
        - the index of the last valid step in the last chunk if right padding.
        - None if no valid step found in the last chunk.
        """
        padding_side = padding_side_dict[self.args.model_name]
        chunk_size = self.args.chunk_size
        if padding_side == "left":
            return chunk_size - 1

        (
            input_chunks,
            _causal_mask_chunks,
            position_ids_chunks,
            _past_key_value_list,
        ) = self.get_prepared_inputs(text_input)
        if not input_chunks or not position_ids_chunks:
            return None

        pos_last = position_ids_chunks[-1]
        if pos_last is None:
            return None
        pos_arr = pos_last.detach().cpu().view(-1).tolist()
        i = len(pos_arr) - 1
        # add pos + 1 to avoid the 0, 1, 1, the first 1 is the valid token
        while i > 0 and pos_arr[i] == 1 and (pos_arr[i - 1] + 1 != pos_arr[i]):
            i -= 1
        return i

    def _get_model_core(self, model_wrapper):
        """Get model core from model wrapper."""
        if model_wrapper is None:
            return None
        return model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    def _create_torch_hook(self, layer_name: str, model_type: str):
        """Create a hook function for Torch model."""

        def hook_fn(module, input, output):
            target_dict = (
                self.torch_vlm_layers_outputs
                if model_type == "vlm"
                else self.torch_layers_outputs
            )
            tensor_info = None
            if isinstance(output, torch.Tensor):
                tensor_info = TensorInfo(output, layer_name)
                target_dict[layer_name] = tensor_info
            elif hasattr(output, "last_hidden_state"):
                tensor_info = TensorInfo(output.last_hidden_state, layer_name)
                target_dict[layer_name] = tensor_info
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    tensor_info = TensorInfo(output[0], layer_name)
                    target_dict[layer_name] = tensor_info

        return hook_fn

    def _register_torch_hooks(self, model, model_type: str):
        """Register hooks for Torch model."""
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(
                    self._create_torch_hook(name, model_type)
                )
                hooks.append(hook)
        return hooks

    def update_kv_cache(
        self,
        past_key_value_list: List[torch.Tensor],
        outputs: dict,
        chunk_size: int,
        transpose_cache: bool = True,
        num_layers: int | None = None,
        convert_from_numpy: bool = False,
    ) -> None:
        """Update KV cache with new outputs.

        Args:
            past_key_value_list: List of past KV tensors to update in-place
            outputs: Dict containing new cache values
            chunk_size: Size of current chunk
            transpose_cache: Whether cache is transposed
            num_layers: Number of model layers (if None, use len(past_key_value_list))
            convert_from_numpy: Whether to convert outputs from numpy
        """
        cache_count = num_layers * 2 if num_layers else len(past_key_value_list)

        for z in range(cache_count):
            if isinstance(outputs, dict):
                output_key = f"_output_{z + 1}"
                if output_key not in outputs:
                    continue
                new_cache = outputs[output_key]
            else:
                new_cache = outputs[z + 1]

            if convert_from_numpy and isinstance(new_cache, np.ndarray):
                new_cache = torch.from_numpy(new_cache)

            past = past_key_value_list[z]

            if hasattr(new_cache, "device") and new_cache.device != past.device:
                new_cache = new_cache.to(past.device)

            if transpose_cache:
                slice_past = past[chunk_size:]
                updated_cache = torch.cat([slice_past, new_cache], dim=0)
            else:
                slice_past = past[:, chunk_size:]
                updated_cache = torch.cat([slice_past, new_cache], dim=-1)

            past_key_value_list[z] = updated_cache

    def prepare_llm_chunk_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value_list: List[torch.Tensor],
        model_meta: list,
        model_name: str,
        torch_llm_model=None,
    ) -> dict:
        """Prepare inputs for LLM inference in a unified way.

        Args:
            input_ids: Token IDs tensor
            position_ids: Position IDs tensor
            attn_mask: Attention mask tensor
            past_key_value_list: List of past KV cache tensors
            model_meta: List of (name, shape, dtype) tuples for model inputs
            model_name: Name of the model
            torch_llm_model: Torch LLM model instance (needed for INTERNVL embeddings)

        Returns:
            Dict of prepared inputs ready for inference
        """
        if not torch_llm_model:
            raise ValueError("Torch LLM model not loaded for embedding.")

        input_dict = {}

        if model_name in INTERNVL_MODELS:
            embedding_layer = getattr(
                torch_llm_model.model, "embed_tokens", None
            ) or getattr(torch_llm_model.model, "tok_embeddings", None)
            if not embedding_layer:
                raise AttributeError("Model lacks embedding layer.")
            with torch.no_grad():
                tok_embs = embedding_layer(input_ids.to(embedding_layer.weight.device))
            input0 = tok_embs.squeeze(0) if tok_embs.dim() == 3 else tok_embs
            input1 = (
                position_ids.squeeze(0) if position_ids.dim() == 2 else position_ids
            )
            input2 = attn_mask.squeeze(0) if attn_mask.dim() == 3 else attn_mask
        else:
            input0, input1, input2 = input_ids, position_ids, attn_mask

        primary_inputs = [input0, input1, input2]
        for i, input_data in enumerate(primary_inputs):
            name, shape, dtype = model_meta[i]
            input_dict[name] = input_data.detach().cpu().numpy().astype(dtype)

        for idx in range(3, len(model_meta)):
            name, _, dtype = model_meta[idx]
            input_data = past_key_value_list[idx - 3]
            input_dict[name] = input_data.detach().cpu().numpy().astype(dtype)

        return input_dict

    def run_llm(self, text: str) -> tuple[TensorDict, TensorDict | None]:
        """Run LLM inference using the appropriate backend based on args."""
        if self.args.compare_mode == "bc":
            return self.run_llm_bc(text)
        elif self.args.compare_mode == "hbm":
            return self.run_llm_hbm(text)
        else:
            return self.run_llm_torch(text)

    def run_vlm(self, image: torch.Tensor) -> tuple[TensorDict, TensorDict | None]:
        """Run VLM inference using the appropriate backend based on args."""
        if self.args.compare_mode == "bc":
            return self.run_vlm_bc(image)
        elif self.args.compare_mode == "hbm":
            return self.run_vlm_hbm(image)
        else:
            return self.run_vlm_torch(image)

    def run_llm_torch(self, text: str) -> tuple[TensorDict, TensorDict]:
        """Run LLM inference using PyTorch backend."""
        self.torch_layers_outputs.clear()
        if self.torch_llm_model_core is None:
            raise ValueError("Torch LLM model core not loaded")

        with time_block("Torch LLM inference (single input)"):
            hooks = self._register_torch_hooks(self.torch_llm_model_core, "llm")
            try:
                outputs = self._run_torch_llm_inference(text)
            finally:
                if hooks:
                    for hook in hooks:
                        hook.remove()
        return outputs, self.torch_layers_outputs

    def run_vlm_torch(self, image: torch.Tensor) -> tuple[TensorDict, TensorDict]:
        """Run VLM inference using PyTorch backend."""
        self.torch_vlm_layers_outputs.clear()
        if self.torch_vlm_model_core is None:
            raise ValueError("Torch VLM model core not loaded")

        hooks = self._register_torch_hooks(self.torch_vlm_model_core, "vlm")
        try:
            with time_block("Torch VLM inference (single input)"):
                outputs = self._run_torch_vlm_inference(image)
        finally:
            if hooks:
                for hook in hooks:
                    hook.remove()
        return outputs, self.torch_vlm_layers_outputs

    def _run_torch_llm_inference(self, text_input: str) -> TensorDict:
        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized.")

        (
            input_chunks,
            causal_mask_chunks,
            position_ids_chunks,
            past_key_value_list,
        ) = self.get_prepared_inputs(text_input)

        if not input_chunks:
            raise ValueError("No input chunks prepared")

        past_key_value_list = [t.to(self.device) for t in past_key_value_list]

        outputs = None
        for idx, (input_ids, attention_mask, position_ids) in enumerate(
            zip(input_chunks, causal_mask_chunks, position_ids_chunks)
        ):
            input_ids = input_ids.to(self.device)
            position_ids = position_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with time_block(f"Torch LLM chunk_{idx}"):
                with torch.no_grad():
                    outputs = self.torch_llm_model_core.forward(
                        input_ids, position_ids, attention_mask, past_key_value_list
                    )

            self.update_kv_cache(
                past_key_value_list,
                outputs,
                chunk_size=input_ids.shape[-1],
                transpose_cache=self.calib_data_preparer.transpose_cache,
                num_layers=self.calib_data_preparer.block_num,
                convert_from_numpy=False,
            )

        if outputs is None:
            raise ValueError("Inference did not produce any output.")

        return cast_to_tensor_info(outputs)

    def _run_torch_vlm_inference(self, image_input: torch.Tensor) -> TensorDict:
        input_to_use = image_input.to(self.device)

        with torch.no_grad():
            outputs = self.torch_vlm_model_core.forward(input_to_use)
        return cast_to_tensor_info(outputs)

    def run_llm_bc(self, text: str) -> tuple[TensorDict, TensorDict]:
        """Run LLM inference using BC backend."""
        self.bc_layers_outputs.clear()
        if not self.bc_model:
            raise ValueError("BC LLM model not loaded")

        self.bc_model.functions[0].register_callback(self._bc_callback)
        with time_block("BC LLM inference (single input)"):
            outputs = self._run_bc_llm_inference(text)
        return cast_to_tensor_info(outputs), self.bc_layers_outputs

    def run_vlm_bc(self, image: torch.Tensor) -> tuple[TensorDict, TensorDict]:
        """Run VLM inference using BC backend."""
        self.bc_vlm_layers_outputs.clear()
        if not self.bc_vlm_model:
            raise ValueError("BC VLM model not loaded")

        self.bc_vlm_model.functions[0].register_callback(self._bc_vlm_callback)

        feed_dict = self._prepare_bc_image_inputs(image)
        with time_block("BC VLM inference"):
            bc_run_outputs = self.bc_vlm_model.functions[0].feed(inputs=feed_dict)
        outputs = OrderedDict([("output", bc_run_outputs["_output_0"])])
        return cast_to_tensor_info(outputs), self.bc_vlm_layers_outputs

    def _bc_callback(self, op, results, operands):
        if op.type == "func.func":
            return True
        if len(results) > 0 and type(results[0]) in [
            torch.Tensor,
            np.ndarray,
            Value,
        ]:
            self.bc_layers_outputs[op.name] = TensorInfo(results[0], op.name)
        return True

    def _bc_vlm_callback(self, op, results, operands):
        if op.type == "func.func":
            return True
        if len(results) > 0 and isinstance(
            results[0], (torch.Tensor, np.ndarray, Value)
        ):
            self.bc_vlm_layers_outputs[op.name] = TensorInfo(results[0], op.name)
        return True

    def _run_bc_llm_inference(self, text_input: str):
        (
            input_chunks,
            causal_mask_chunks,
            position_ids_chunks,
            past_key_value_list,
        ) = self.get_prepared_inputs(text_input)

        if not self.bc_model:
            raise ValueError("BC model is not loaded")

        for idx, (input_ids, attn_mask, position_ids) in enumerate(
            zip(input_chunks, causal_mask_chunks, position_ids_chunks)
        ):
            feed_dict = self._prepare_bc_chunk_inputs(
                input_ids, position_ids, attn_mask, past_key_value_list
            )

            with time_block(f"BC LLM chunk_{idx}"):
                outputs = self.bc_model.functions[0].feed(inputs=feed_dict)

            self.update_kv_cache(
                past_key_value_list,
                outputs,
                chunk_size=self.args.chunk_size,
                transpose_cache=True,  # BC always slices on first dimension
                num_layers=None,  # Use len(past_key_value_list)
                convert_from_numpy=True,
            )
        return outputs

    def _prepare_bc_chunk_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value_list: List[torch.Tensor],
    ) -> dict:
        if not self.bc_model:
            raise ValueError("BC model not loaded")

        bc_inputs_meta = [
            (inp.name, inp.type.shape, inp.type.np_dtype)
            for inp in self.bc_model.functions[0].inputs
        ]

        return self.prepare_llm_chunk_inputs(
            input_ids,
            position_ids,
            attn_mask,
            past_key_value_list,
            bc_inputs_meta,
            self.args.model_name,
            self.torch_llm_model,
        )

    def _prepare_bc_image_inputs(self, image_input: torch.Tensor) -> dict:
        """Prepare image inputs for BC VLM model."""
        if not self.bc_vlm_model:
            raise ValueError("BC VLM model is not loaded")
        if image_input.dim() != 4:
            raise ValueError(
                f"Expected 4D image input (N,C,H,W), got {image_input.dim()}D"
            )
        dtype = self.bc_vlm_model.functions[0].inputs[0].type.np_dtype
        return {"_input_0": image_input.detach().cpu().numpy().astype(dtype)}

    def run_llm_hbm(self, text: str) -> tuple[TensorDict, None]:
        """Run LLM inference using HBM backend."""
        if not self.args.hbm_llm_model_path:
            raise ValueError("HBM model path not specified")
        with time_block("HBM LLM inference (single input)"):
            outputs = self._run_hbm_llm_inference(text)
        return outputs, None

    def run_vlm_hbm(self, image: torch.Tensor) -> tuple[TensorDict, None]:
        """Run VLM inference using HBM backend."""
        if not self.args.hbm_vlm_model_path:
            raise ValueError("HBM VLM model path not specified")
        with time_block("HBM VLM inference (single input)"):
            outputs = self._run_hbm_vlm_inference(image)
        return outputs, None

    def _run_hbm_llm_inference(self, text_input: str) -> TensorDict:
        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized")

        if self.torch_llm_model is None:
            raise ValueError("Torch LLM model not loaded")

        (
            input_chunks,
            causal_mask_chunks,
            position_ids_chunks,
            past_key_value_list,
        ) = self.get_prepared_inputs(text_input)

        if not input_chunks:
            raise ValueError("No input chunks prepared")

        model_args = self.torch_llm_model.get_model_args()

        if self._hbm_llm_module is None:
            raise ValueError("HBM LLM module not loaded")

        hbm_inputs_meta = [
            (inp.name, inp.type.shape, inp.type.np_dtype)
            for inp in self._hbm_llm_module.inputs
        ]

        outputs = None
        for idx, (input_ids, attn_mask, position_ids) in enumerate(
            zip(input_chunks, causal_mask_chunks, position_ids_chunks)
        ):
            model_inputs = self.prepare_llm_chunk_inputs(
                input_ids,
                position_ids,
                attn_mask,
                past_key_value_list,
                hbm_inputs_meta,
                self.args.model_name,
                self.torch_llm_model,
            )

            res = self._get_hbm_infer_res(self._hbm_llm_module, model_inputs)
            outputs = res

            self.update_kv_cache(
                past_key_value_list,
                res,
                chunk_size=input_ids.shape[-1],
                transpose_cache=self.calib_data_preparer.transpose_cache,
                num_layers=model_args.num_hidden_layers,
                convert_from_numpy=True,
            )

        if outputs is None:
            raise ValueError("HBM inference did not produce any output.")

        return OrderedDict({k: TensorInfo(v, name=k) for k, v in outputs.items()})

    def _run_hbm_vlm_inference(self, image_input: torch.Tensor) -> TensorDict:
        if self._hbm_vlm_module is None:
            raise RuntimeError("HBM VLM module not loaded. Check configuration.")

        model_inputs = {"_input_0": image_input.cpu().type(torch.float16).numpy()}
        res = self._get_hbm_infer_res(self._hbm_vlm_module, model_inputs)
        return OrderedDict([(k, TensorInfo(v, name=k)) for k, v in res.items()])

    def _get_hbm_infer_res(self, hbm_module: Graph, model_inputs: dict) -> dict:
        """Execute inference on a *pre-loaded* HBM Graph."""
        if not self.args.remote_ip or not self.args.remote_path:
            raise ValueError("Remote IP and path must be specified for HBM inference")
        try:
            if os.environ.get("LLM_VERIFIER_DEBUG"):
                print(f"HBM module: {hbm_module}")
                print(f"Model inputs: {model_inputs}")
                print(f"Remote IP: {self.args.remote_ip}")
                print(f"Username: {self.args.username}")
                print(f"Port: {self.args.port}")
                print(f"Password: {self.args.password}")
                print(f"Remote work root: {self.args.remote_path}")
            res = hbm_module.feed(
                feed_dict=model_inputs,
                remote_ip=self.args.remote_ip,
                username=self.args.username,
                remote_port=self.args.port,
                password=self.args.password,
                remote_work_root=self.args.remote_path,
            )
            return res
        except Exception as exc:
            msg = (
                "Failed to perform inference on remote host "
                f"{self.args.remote_ip}: {exc}"
            )
            raise ConnectionError(msg)
