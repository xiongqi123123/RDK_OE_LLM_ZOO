import os
from pathlib import Path

import torch
from hbdk4.compiler import leap
from safetensors import safe_open

from leap_llm.apis.calibration.data_loader import load_image_data
from leap_llm.models.siglip.model import SiglipVision


def _load_siglip_image_data(calib_image_path=None, image_size=384):
    """Load and preprocess calibration images for SigLIP.

    SigLIP uses simple resize to (image_size, image_size) with
    mean=0.5, std=0.5 normalization (range [-1, 1]).
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if calib_image_path is None:
        # Use default calibration images from leap_llm
        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "calibration", "calibration_data", "images",
        )
        if os.path.isdir(default_dir):
            calib_image_path = default_dir
        else:
            print("Warning: No default calibration images found.")
            return

    if os.path.isfile(calib_image_path):
        image_files = [calib_image_path]
    elif os.path.isdir(calib_image_path):
        exts = {".jpg", ".jpeg", ".png"}
        image_files = sorted([
            os.path.join(calib_image_path, f)
            for f in os.listdir(calib_image_path)
            if os.path.splitext(f)[1].lower() in exts
        ])
    else:
        print(f"Warning: calib_image_path not found: {calib_image_path}")
        return

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0)  # (1, 3, H, W)
        yield pixel_values


class SiglipApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_image_path: str = None,
        device: str = "cpu",
        model_type: str = "siglip-so400m",
        core_num: int = 1,
    ):
        self.input_model_path = input_model_path
        self.device = device
        self.core_num = core_num

        os.makedirs(output_model_path, exist_ok=True)
        self.vit_file_name = os.path.join(
            output_model_path, f"{model_type}_vit_ptq.hbm"
        )

        # Load checkpoint
        checkpoint = {}
        safetensors_path = os.path.join(input_model_path, "model.safetensors")
        pytorch_path = os.path.join(input_model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        elif os.path.exists(pytorch_path):
            checkpoint = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model weights found in {input_model_path}. "
                "Expected model.safetensors or pytorch_model.bin."
            )

        self.vit_model = SiglipVision.load_model(input_model_path, checkpoint)

        # Load image calibration data with SigLIP preprocessing
        self.calib_image_data = list(
            _load_siglip_image_data(
                calib_image_path,
                image_size=self.vit_model.config.image_size,
            )
        )
        print(f"Loaded {len(self.calib_image_data)} calibration images.")

    def compile(self, **kwargs):
        device = self.device if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        vit_module = self.vit_model
        vit_module.set_model_device(device, dtype=dtype)
        vit_module.set_compile_mode(False)

        # Calibration: forward pass to collect quantization statistics
        print("Running calibration on vision model...")
        for i, image_pixel in enumerate(self.calib_image_data):
            vit_module.forward(image_pixel.to(device))
            print(f"  Calibrated image {i + 1}/{len(self.calib_image_data)}")

        # Compile
        print("Compiling vision model...")
        vit_module.set_model_device("cpu", dtype=torch.float16)
        vit_module.set_compile_mode(True)
        vit_module.compile(
            dtype=leap.float16,
            output_model_path=self.vit_file_name,
            core_num=self.core_num,
            **kwargs,
        )
        print(f"Compiled model saved to: {self.vit_file_name}")

    def get_hbm_path(self) -> tuple:
        return (self.vit_file_name,)
