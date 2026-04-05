_model_builders = {}


def register_model(name, marches=None):
    def decorator(func):
        _model_builders[name] = {"builder": func, "marches": marches or []}
        return func

    return decorator


def get_supported_models():
    return list(_model_builders.keys())


def get_marches_with_model(model_name: str) -> list[str]:
    return _model_builders.get(model_name, {}).get("marches", [])


def get_supported_marches():
    return list(
        set(
            march
            for model_name in _model_builders.keys()
            for march in _model_builders[model_name]["marches"]
        )
    )


def create_model_api(model_name, args):
    model_info = _model_builders.get(model_name)
    if not model_info:
        print(f"Model '{model_name}' is not supported yet.")
        return None

    supported_marches = get_supported_marches()
    if args.march not in supported_marches:
        print(f"March {args.march} is not supported for model {model_name}.")
        print(f"Supported marches are: {', '.join(supported_marches)}")
        return None

    builder = model_info["builder"]
    return builder(args)


@register_model("deepseek-qwen-1_5b", ["nash-e", "nash-m"])
def _build_deepseek_qwen_1_5b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    preserve_precision = False
    mask_value = -512
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        preserve_precision=preserve_precision,
        model_type="deepseek-qwen-1_5b",
        w_bits=args.w_bits,
        mask_value=mask_value,
    )


@register_model("deepseek-qwen-7b", ["nash-e", "nash-m"])
def _build_deepseek_qwen_7b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    preserve_precision = True
    dtype = "float16"
    mask_value = -512
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        dtype=dtype,
        preserve_precision=preserve_precision,
        model_type="deepseek-qwen-7b",
        mask_value=mask_value,
    )


@register_model("qwen2_5-1_5b", ["nash-e", "nash-m"])
def _build_qwen2_5_1_5b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    preserve_precision = True
    mask_value = -32767
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        preserve_precision=preserve_precision,
        model_type="qwen2_5-1_5b",
        mask_value=mask_value,
    )


@register_model("qwen2_5-7b", ["nash-e", "nash-m"])
def _build_qwen2_5_7b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    dtype = "float16"
    preserve_precision = True
    mask_value = -16384
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        dtype=dtype,
        preserve_precision=preserve_precision,
        model_type="qwen2_5-7b",
        mask_value=mask_value,
    )


@register_model("internvl2-2b", ["nash-p"])
def _build_internvl2_2b(args):
    from leap_llm.apis.model.internvl_2b import Internvl2bApi

    return Internvl2bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        vlm_model_type="internvl2-2b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("internvl2_5-2b", ["nash-p"])
def _build_internvl2_5_2b(args):
    from leap_llm.apis.model.internvl_2b import Internvl2bApi

    return Internvl2bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        vlm_model_type="internvl2_5-2b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("internvl2-1b", ["nash-p"])
def _build_internvl2_1b(args):
    from leap_llm.apis.model.internvl_1b import Internvl1bApi

    return Internvl1bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        vlm_model_type="internvl2-1b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("internvl2_5-1b", ["nash-p"])
def _build_internvl2_5_1b(args):
    from leap_llm.apis.model.internvl_1b import Internvl1bApi

    return Internvl1bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        vlm_model_type="internvl2_5-1b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("internlm2-1_8b", ["nash-e", "nash-m"])
def _build_internlm2_18b(args):
    from leap_llm.apis.model.internlm2 import Internlm2Api

    return Internlm2Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        dtype="float32",
        preserve_precision=False,
        model_type="internlm2-1_8b",
    )


@register_model("siglip-so400m", ["nash-e", "nash-m", "nash-p"])
def _build_siglip_so400m(args):
    from leap_llm.apis.model.siglip import SiglipApi

    return SiglipApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        device=args.device,
        model_type="siglip-so400m",
        core_num=args.vit_core_num,
    )


@register_model("gemma4-e2b-vision", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e2b_vision(args):
    from leap_llm.apis.model.gemma4 import Gemma4VisionApi

    return Gemma4VisionApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        device=args.device,
        model_type="gemma4-e2b",
        core_num=args.vit_core_num,
    )


@register_model("gemma4-e2b-text", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e2b_text(args):
    from leap_llm.apis.model.gemma4 import Gemma4TextApi

    return Gemma4TextApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        model_type="gemma4-e2b",
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("qwen2_5-omni-3b", ["nash-e", "nash-m"])
def _build_qwen2_5_omni_3b(args):
    from leap_llm.apis.model.qwen2_5_omni import Qwen2_5OmniApi

    if args.chunk_size != 256 or args.cache_len != 2048:
        print(
            f"Warning: {args.model_name} model only supports chunk_size=256 and "
            f"cache_len=2048."
            "Setting chunk_size to 256 and cache_len to 2048."
        )
        args.chunk_size = 256
        args.cache_len = 2048

    return Qwen2_5OmniApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_conversation_path=args.calib_conversation_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        dtype="float32",
        preserve_precision=True,
        model_type="qwen2_5_omni_3b",
    )
