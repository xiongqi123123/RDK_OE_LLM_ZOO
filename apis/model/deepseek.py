import os
import shutil
from pathlib import Path

import torch

from leap_llm.apis.calibration.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import load_text_data
from leap_llm.models.deepseek.model import DeepSeek


class DeepSeekApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 512,
        device: str = "cpu",
        dtype: str = "float32",
        preserve_precision: bool = False,
        model_type: str = "deepseek",
        w_bits: int = 8,
        mask_value: int = -512,
    ):
        self.input_model_path = input_model_path
        self.calib_text_data = load_text_data(calib_text_path)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = dtype
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.model_type = model_type

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_path = os.path.join(
            output_model_path,
            f"{model_type}_chunk_{chunk_size}_cache_{cache_len}_q{w_bits}.hbm",
        )

        self.deepseek_model = DeepSeek.build(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
            preserve_precision=preserve_precision,
            w_bits=w_bits,
        )

    def compile(self, **kwargs):
        device = self.device if torch.cuda.is_available() else "cpu"

        if "7b" in self.model_type:
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.deepseek_model.model.to(device, dtype=dtype)
        self.deepseek_model.model.compile_mode(False)

        transpose_cache = True
        preparer = CalibrationDataPreparer(
            self.input_model_path,
            self.chunk_size,
            self.cache_len,
            transpose_cache=transpose_cache,
            device=device,
            mask_value=self.mask_value,
        )
        # set the padding_side to left on tokenizer
        preparer.tokenizer.padding_side = "left"

        for prompt in self.calib_text_data:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                pask_key_value_list,
            ) = preparer.prepare_inputs(prompt)

            for i, (input_ids, attn_mask, position_ids) in enumerate(
                zip(input_chunks, causal_mask_chunks, position_ids_chunks)
            ):
                with torch.no_grad():
                    outputs = self.deepseek_model.model.forward(
                        input_ids, position_ids, attn_mask, pask_key_value_list
                    )

                for z in range(0, self.deepseek_model.model_args.num_hidden_layers * 2):
                    new_cache = outputs[z + 1]
                    past = pask_key_value_list[z]

                    if transpose_cache:
                        slice_past = past[self.chunk_size :, :, :]
                    else:
                        slice_past = past[:, self.chunk_size :, :]

                    dim = 0 if transpose_cache else -2
                    update_cache = torch.concat([slice_past, new_cache], dim=dim)
                    pask_key_value_list[z] = update_cache

        self.deepseek_model.model.compile_mode(True)
        self.deepseek_model.model.to("cpu")

        self.deepseek_model.compile(
            stage="all",
            output_model_path=self.output_model_path,
            **kwargs,
        )

    def get_quant_path(self) -> tuple[str, None]:
        """Return fixed DeepSeek BC path."""

        return str(
            Path(self.output_model_path).with_suffix(".prefill_convert_removed.bc")
        ), None

    def get_hbm_path(self) -> tuple[str, None]:
        """Return fixed DeepSeek HBM path."""

        return self.output_model_path, None
