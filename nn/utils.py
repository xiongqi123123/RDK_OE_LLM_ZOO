import inspect
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypeVar

import torch
from hbdk4.compiler import compile, convert, leap, link, save, statistics
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
from hbdk4.compiler.dialects.hbir import TrackAttr
from hbdk4.compiler.extra_apis import llm_convert
from hbdk4.compiler.hbm import Hbm, Hbo
from safetensors import safe_open
from torch import nn
from torch.nn import ModuleList

T = TypeVar("T", bound="Module")


def timeit(func):
    """
    This decorator prints the execution time of the decorated function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' done in {execution_time:.4f} seconds.")
        return result

    return wrapper


@contextmanager
def timeit_context(name: str = "block"):
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Block '{name}' done in {execution_time:.4f} seconds.")


class ModuleContainter:
    """Use container to avoid torch recorded as named_xxx"""

    def __init__(self, module: "Module"):
        self.module = module


class ModuleMeta(type):
    def __call__(cls, *args, **kwargs):
        instance: Module = super().__call__(*args, **kwargs)

        instance.set_children_parent()
        instance.set_forward_location()

        orig_forward = instance.forward

        def new_forward(*args, **kwargs):
            if instance.is_compiled:
                name = instance.get_full_name()
                with mlir.Location.name(name):
                    return instance.build(*args, **kwargs)
            else:
                with torch.no_grad():
                    return orig_forward(*args, **kwargs)

        instance.forward = new_forward
        return instance


class Module(nn.Module, metaclass=ModuleMeta):
    """Alias of torch.nn.Module. For now."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled = True  # 初始化 _compiled 标志
        if not hasattr(self, "_parent"):
            self._parent: Optional[ModuleContainter] = None

    @property
    def is_compiled(self):
        return self._compiled

    def compile_mode(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("compile_mode mode is expected to be boolean")

        self._compiled = mode

        for child in self.children():
            if isinstance(child, ModuleList):
                for subchild in child:
                    subchild.compile_mode(mode)
            elif isinstance(child, Module):
                child.compile_mode(mode)

    @property
    def name(self) -> str:
        # each time you access .name it calls your get_full_name logic
        return self.get_full_name()

    def get_full_name(self) -> str:
        node = self
        parent = getattr(node, "_parent", None)
        while parent is not None:
            node = parent.module if hasattr(parent, "module") else parent
            parent = getattr(node, "_parent", None)

        root = node

        for full_name, mod in root.named_modules():
            if mod is self:
                return full_name

        return self.__class__.__name__

    def set_children_parent(self):
        """
        After __init__
        """
        for _, c in self.named_modules():
            if getattr(c, "_parent", None) is None and c is not self:
                c._parent = ModuleContainter(self)

    def _wrap_with_location(self, func):
        def wrapped_func(*args, **kwargs):
            with self.markLeafOPwith(self):
                return func(*args, **kwargs)

        return wrapped_func

    def set_forward_location(self):
        self._original_forward = self.forward
        self.forward = self._wrap_with_location(self.forward)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def _model(self) -> "Module":
        if self._parent:
            return self._parent.module._model
        return self

    def get_module_local_name(self, module):
        for name, m in self.named_modules():
            if m is module:
                return name
        return None

    def get_module_global_name(self, param):
        return self._model.get_module_local_name(param)

    def markLeafOPwith(self, m: "Module"):
        forward_func = (
            self._original_forward
            if hasattr(self, "_original_forward") and callable(self._original_forward)
            else self.forward
        )

        filename = inspect.getsourcefile(forward_func)
        if filename:
            line = inspect.getsourcelines(forward_func)[1]
            loc = mlir.Location.file(
                filename=filename,
                line=line,
                col=0,
                context=get_default_loc_context(),
            )
        else:
            loc = mlir.Location.unknown()
        loc = mlir.Location.fused(
            [loc],
            TrackAttr.get(dict(module_name=self.get_module_global_name(m))),
        )
        return loc


class Model(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_forward_module(
        self,
        input_types: List[leap.TensorType],
        name: Optional[str] = None,
        print=False,
    ):
        leap_func = leap.leap_export(self.forward, *input_types, name=name)
        if print:
            leap_func.print(enable_debug_info=True, large_elements_limit=16)
        return leap_func.module

    @timeit
    def export_module(
        self,
        input_types: List[leap.TensorType],
        name: Optional[str] = None,
        save_path: Optional[str] = None,
        high_precision_qpp: bool = False,
    ):
        bc_module = self.get_forward_module(input_types, name)
        bc_module._llm_extra = True
        bc_module._high_precision_qpp = high_precision_qpp
        if save_path:
            save(bc_module, save_path)
        return bc_module

    @staticmethod
    @timeit
    def convert_mlir(
        dq_bc_module,
        save_path: Optional[str] = None,
        enable_vpu=True,
        march="nash-m",
        dynamic_quant=False,
    ):
        if dynamic_quant:
            dq_bc_module = llm_convert(dq_bc_module, march, rmsnorm_version="cuda")

        if march == "nash-p":
            dq_bc_module._use_f16_quant_dequant_on_vae_always = True

        mlir_module = convert(dq_bc_module, march, enable_vpu=enable_vpu)
        statistics(mlir_module)
        if save_path:
            save(mlir_module, save_path)
        return mlir_module

    @staticmethod
    @timeit
    def compile_hbo(mlir_module, save_path: str, **kwargs) -> Hbo:
        assert save_path.endswith(".hbo"), (
            f"save_path must end with .hbo, but got {save_path}"
        )

        hbo_model = compile(
            mlir_module,
            save_path,
            **kwargs,
        )
        return hbo_model

    @staticmethod
    @timeit
    def link_models(hbo_list: List[Hbo], save_path: str) -> Hbm:
        assert save_path.endswith(".hbm")
        return link(hbo_list, save_path)

    @timeit
    def compile_model(
        self,
        input_types: List[leap.TensorType],
        compile_args: dict,
        quantize_args: dict,
        output_hbm_path: str,
    ) -> Hbm:
        bc_module = self.export_module(
            input_types,
            compile_args.get("name"),
            str(Path(output_hbm_path).with_suffix(".bc")),
        )

        dq_bc_module = self.dynamic_quantize(
            bc_module,
            quantize_args,
            str(Path(output_hbm_path).with_suffix(".dynamic_quantize.bc")),
        )

        mlir_module = self.convert_mlir(dq_bc_module)
        hbo_model = self.compile_hbo(
            mlir_module,
            str(Path(output_hbm_path).with_suffix(".hbo")),
            **compile_args,
        )
        hbm_model = self.link_models([hbo_model], output_hbm_path)
        return hbm_model


def load_safetensors_state_dict(
    model_dir: str,
    include_substrings: list[str] = ["weight", "bias", "buf_scales"],
    content_change_map: Dict[str, str] = {".buf_scales": ".scales"},
    prefix_remove_list: list[str] = ["model."],
) -> Dict[str, torch.Tensor]:
    """Load the state_dict from the model directory.

    Args:
        model_dir (str): The directory of the model.
        include_substrings (Sequence[str], optional): The substrings of the model.
        content_change_map (Dict[str, str], optional):
            The content change map of the model.
        prefix_remove_list (list[str], optional):
            The prefix change map of the model.

    Returns:
        Dict[str, torch.Tensor]: The state_dict of the model.
    """
    assert os.path.isdir(model_dir), f"'{model_dir}' is not a valid directory"

    shard_files = sorted(
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    )

    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    state: Dict[str, torch.Tensor] = {}
    for shard_path in shard_files:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if include_substrings and not any(s in key for s in include_substrings):
                    continue

                new_key = key
                for pattern, replacement in content_change_map.items():
                    if pattern in new_key:
                        new_key = new_key.replace(pattern, replacement)

                for prefix in prefix_remove_list:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]

                state[new_key] = f.get_tensor(key)

    return state
