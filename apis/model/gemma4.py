import os
from pathlib import Path

import torch
from hbdk4.compiler import leap
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from leap_llm.models.gemma4.model import Gemma4Vision, Gemma4VisionConfig


def _patchify_image(image_tensor, patch_size=16):
    """Convert image tensor [1, 3, H, W] -> [1, num_patches, 3*patch_size^2].

    Extracts non-overlapping patches in row-major order and flattens each patch.
    """
    _, C, H, W = image_tensor.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    # [1, 3, h_patches, patch_size, w_patches, patch_size]
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [1, h_patches, w_patches, 3, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [1, num_patches, 3*patch_size^2]
    patches = patches.view(1, h_patches * w_patches, C * patch_size * patch_size)
    return patches


def _load_gemma4_image_data(image_path, h_patches=48, w_patches=48, patch_size=16):
    """Load and preprocess images for Gemma4 vision encoder.

    Resizes to (h_patches*patch_size, w_patches*patch_size), normalizes to [0, 1],
    then patchifies.

    Yields: [1, num_patches, 3*patch_size^2] tensors.
    """
    target_h = h_patches * patch_size
    target_w = w_patches * patch_size

    transform = transforms.Compose([
        transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
    ])

    if image_path is None:
        image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "calibration", "calibration_data", "images",
        )

    if os.path.isdir(image_path):
        image_files = sorted(
            p for p in Path(image_path).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    else:
        image_files = [Path(image_path)]

    for img_path in image_files:
        img = Image.open(str(img_path)).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
        patches = _patchify_image(tensor, patch_size)  # [1, num_patches, patch_dim]
        yield patches


class Gemma4VisionApi:
    """API for Gemma4 vision encoder quantization and compilation."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_image_path: str = None,
        device: str = "cpu",
        model_type: str = "gemma4-e2b",
        core_num: int = 1,
    ):
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.calib_image_path = calib_image_path
        self.device = device
        self.model_type = model_type
        self.core_num = core_num

        os.makedirs(output_model_path, exist_ok=True)
        self.vit_file_name = os.path.join(
            output_model_path, f"{model_type}_vit_ptq.hbm"
        )

        # Load model
        self.vit_model = Gemma4Vision.load_model(input_model_path)
        self.config = self.vit_model.config

        # Pre-load calibration images
        self.calib_images = list(
            _load_gemma4_image_data(
                calib_image_path,
                h_patches=self.config.h_patches,
                w_patches=self.config.w_patches,
                patch_size=self.config.patch_size,
            )
        )
        print(f"[Gemma4VisionApi] Loaded {len(self.calib_images)} calibration images")

    def compile(self, **kwargs):
        device = self.device if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        vit = self.vit_model

        # Calibration: forward pass to collect statistics
        vit.set_model_device(device, dtype=dtype)
        vit.set_compile_mode(False)

        print("[Gemma4VisionApi] Running calibration...")
        for patches in tqdm(self.calib_images, desc="Calibrating"):
            vit.forward(patches.to(device, dtype=dtype))

        # Compile
        vit.set_model_device("cpu", dtype=torch.float32)
        vit.set_compile_mode(True)
        vit.compile(
            dtype=leap.float16,
            output_model_path=self.vit_file_name,
            vit_core_num=self.core_num,
            **kwargs,
        )

    def get_hbm_path(self):
        return (self.vit_file_name,)
