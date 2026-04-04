import json
import math
import os
from typing import Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


def pad_to_multiple(n: int, multiple: int = 256) -> Tuple[int, int]:
    """
    Compute the smallest multiple of 'multiple' that is greater than or equal to n,
    and the amount required to pad n to that multiple.

    Args:
        n: The original value.
        multiple: The factor to multiply.

    Returns:
        rounded: The smallest multiple of 'multiple' that is >= n.
        padding: The number needed to pad n.
    """
    rounded = math.ceil(n / multiple) * multiple
    padding = rounded - n
    return rounded, padding


def create_chunk_mask(
    causal_mask_chunk: torch.Tensor,
    chunk_size: int = 256,
    mask_value: float = -2048,
    kv_cache_len: int = 512,
    device="cpu",
) -> torch.Tensor:
    """
    Create a chunk mask based on the input causal_mask_chunk.

    Args:
        causal_mask_chunk: The input causal mask chunk tensor.
        chunk_size: The size of each chunk.
        mask_value: The fill value for the mask.
        kv_cache_len: The length of the KV cache.

    Returns:
        The concatenated chunk mask tensor.
    """
    causal_mask_chunk_slice = causal_mask_chunk[:, :, -1:, :]
    tensor_1d = causal_mask_chunk_slice[0, 0, 0, :]
    zero_indices = (tensor_1d == 0).nonzero(as_tuple=False).squeeze()

    if zero_indices.dim() == 0:
        zero_indices = zero_indices.unsqueeze(0)

    last_zero_index = zero_indices[-1].item() if zero_indices.numel() > 0 else None
    if last_zero_index is None:
        # If no zero found, keep the full range
        last_zero_index = causal_mask_chunk.shape[-1] - 1
    slice_chunk_mask = causal_mask_chunk[:, :, :, : last_zero_index + 1]
    slice_chunk_mask_len = slice_chunk_mask.shape[-1]
    init_zeros = torch.zeros([1, 1, chunk_size, kv_cache_len - slice_chunk_mask_len])
    pad_mask = init_zeros + mask_value
    pad_mask = pad_mask.to(device)
    concat_chunk_mask = torch.cat((pad_mask, slice_chunk_mask), dim=-1)
    return concat_chunk_mask


def get_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute position_ids by cumulatively summing the attention mask.
    Positions corresponding to actual tokens (attention_mask == 1)
    are assigned cumulative positions, and positions for padding
    (attention_mask == 0) are set to 1.

    Args:
        attention_mask: The attention mask tensor.

    Returns:
        A tensor of position_ids.
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Construct a 4D causal attention mask of shape
    (batch_size, 1, query_length, key_value_length).

    If the input attention_mask is already 4D, it is returned directly.

    Args:
        attention_mask: A tensor with shape (batch_size, key_value_length)
            or (batch_size, 1, query_length, key_value_length).
        sequence_length: The length of the input sequence.
        target_length: The target length. When using a static cache,
            the mask length should match the cache.
        dtype: The data type of the constructed mask.
        device: The device on which the mask is placed.
        min_dtype: The minimum representable value for the given dtype.
        cache_position: Tensor indicating the positions of input tokens in the sequence.
        batch_size: Batch size.

    Returns:
        The constructed 4D causal mask.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # If the mask is already 4D, return it directly.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device).reshape(
            1, -1
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # Clone to ensure contiguous memory for in-place modifications.
            mask_length = attention_mask.shape[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            ) == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def update_causal_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    min_dtype: float = -3.4028234663852886e38,
    sequence_length: int = 2048,
    kv_cache_len: int = 2048,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Update and generate a 4D causal mask based on a 2D attention mask.

    Args:
        attention_mask: Input attention mask tensor.
        input_tensor: Model input tensor used to obtain the batch_size.
        cache_position: Tensor indicating token positions.
        min_dtype: The minimum value for the given dtype.
        sequence_length: The sequence length.
        kv_cache_len: The KV cache length.
        dtype: The data type.
        device: The device information.

    Returns:
        Updated 4D causal mask.
    """
    target_length = kv_cache_len
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )
    return causal_mask


class CalibrationDataPreparer:
    def __init__(
        self,
        model_dir: str,
        seq_len: int,
        kv_cache_len: int,
        transpose_cache: bool = True,
        device: str = "cpu",
        mask_value: float = -512,
    ):
        """
        初始化阶段加载 tokenizer 及模型配置

        参数:
          model_dir: 模型及配置所在的目录
          seq_len: 分块的序列长度
          kv_cache_len: kv 缓存长度
          transpose_cache: 是否转置缓存
          device: 设备，默认为 "cpu"
          mask_value: 注意力掩码的填充值，默认为 -512
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.seq_len = seq_len
        self.kv_cache_len = kv_cache_len
        self.transpose_cache = transpose_cache

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "llm_config" in config.keys():
            config_dict = config["llm_config"]
        else:
            config_dict = config
        self.hidden_size = config_dict["hidden_size"]
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.head_dim = int(self.hidden_size / self.num_attention_heads)
        self.num_key_value_heads = config_dict["num_key_value_heads"]
        self.block_num = config_dict["num_hidden_layers"]
        self.mask_value = mask_value

    def prepare_inputs(self, prompt: str):
        """
        处理单个 prompt，返回 input_chunks、causal_mask_chunks、
        position_ids_chunks、past_key_values_list

        参数:
          prompt: 待处理的文本 prompt
        """
        # 对 prompt 进行初步 tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids_valid = inputs.input_ids.to(self.device)
        raw_inputs_len = input_ids_valid.shape[-1]

        if raw_inputs_len > self.kv_cache_len:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.kv_cache_len - 10,
            )
            input_ids_valid = inputs.input_ids.to(self.device)

        n = input_ids_valid.shape[-1]
        inputs_pad_len, _ = pad_to_multiple(n, self.seq_len)

        chunk_size = self.seq_len
        max_length = inputs_pad_len

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = get_position_ids(attention_mask).to(self.device)

        # 按 chunk_size 拆分 input_ids 和 position_ids
        input_chunks = input_ids.split(chunk_size, dim=-1)
        position_ids_chunks = position_ids[0].split(chunk_size, dim=-1)

        # 构造注意力 mask，用于 kv cache，假设最后 valid_seq_len 个位置是有效的
        valid_seq_len = int(torch.sum(attention_mask).item())
        attention_mask_np = np.zeros((1, self.kv_cache_len), dtype=np.int64)
        attention_mask_np[0, -valid_seq_len:] = 1
        attention_mask_tensor = torch.from_numpy(attention_mask_np).to(self.device)
        cache_position = torch.arange(0, self.kv_cache_len, device=self.device)

        causal_mask = update_causal_mask(
            attention_mask_tensor,
            input_ids,
            cache_position,
            min_dtype=self.mask_value,
            sequence_length=self.kv_cache_len,
            kv_cache_len=self.kv_cache_len,
            dtype=torch.float32,
            device=str(self.device),
        )
        # 保留最新部分的 causal mask
        causal_mask = causal_mask[:, :, -input_ids.shape[1] :, :]

        # 初始化 past key/value（KV）缓存数据
        init_kv_data = torch.zeros(
            [self.num_key_value_heads, self.kv_cache_len, self.head_dim],
            dtype=torch.float32,
        ).to(self.device)

        if self.transpose_cache:
            init_kv_data = init_kv_data.transpose(0, 1)

        past_key_values_list: List[Any] = [init_kv_data] * self.block_num + [
            init_kv_data
        ] * self.block_num

        # 将 causal mask 拆分成 chunks
        causal_mask_chunks = causal_mask.split(chunk_size, dim=-2)

        update_causal_mask_chunks = []
        for causal_chunk in causal_mask_chunks:
            chunk_mask = create_chunk_mask(
                causal_chunk,
                chunk_size=chunk_size,
                mask_value=self.mask_value,
                kv_cache_len=self.kv_cache_len,
                device=self.device,
            )
            # Squeeze extra dimensions.
            if chunk_mask.ndim == 4:
                chunk_mask = chunk_mask.squeeze(0).squeeze(0)
            update_causal_mask_chunks.append(chunk_mask.to(self.device))

        return (
            input_chunks,
            update_causal_mask_chunks,
            position_ids_chunks,
            past_key_values_list,
        )
