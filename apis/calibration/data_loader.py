from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import torch

from leap_llm.apis.calibration.image_process import load_image

__all__ = ["load_text_data", "load_image_data", "load_conversation_data"]


# set default calibration data path
DEFAULT_PROMPTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibration_data",
    "texts",
    "prompts.json",
)

DEFAULT_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibration_data",
    "images",
)

# set supported image file types
SUPPORTED_IMAGE_FILE_TYPES = [".jpg", ".jpeg", ".png"]

# Default conversation data path
DEFAULT_CONVERSATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibration_data",
    "omni",
    "conversation.json",
)


def _normalize_media_paths(conversation: list, json_dir: Path) -> list:
    """Normalize media paths to absolute paths in-memory.

    Args:
        conversation (list): The conversation data to normalize.
        json_dir (Path): The directory containing the JSON file.

    Returns:
        list: The normalized conversation data.
    """
    repo_root = Path(__file__).resolve().parents[3]
    base_candidates: list[Path] = [json_dir, repo_root]

    media_keys = ("audio", "image", "video")
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for element in content:
            for key in media_keys:
                value = element.get(key)
                if not value:
                    continue
                raw = Path(os.path.expanduser(str(value)))
                if raw.is_absolute():
                    element[key] = str(raw)
                    continue

                resolved: Path | None = None
                for base in base_candidates:
                    candidate = (base / raw).resolve()
                    if candidate.exists():
                        resolved = candidate
                        break

                if resolved is None:
                    resolved = (json_dir / raw).resolve()

                element[key] = str(resolved)

    return conversation


def _validate_prompt_item(item: dict, source: str) -> str:
    if not isinstance(item, dict):
        raise ValueError(
            f"Invalid prompt entry in {source}: each item must be an object."
        )
    if "text" not in item:
        raise ValueError(f"Invalid prompt entry in {source}: missing 'text' field.")
    text = item["text"]
    if not isinstance(text, str):
        raise ValueError(f"Invalid prompt entry in {source}: 'text' must be a string.")
    return text


def _iter_prompts_from_file(json_path: str) -> Iterator[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse JSON file {json_path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(
            f"Prompt JSON root must be a list, got {type(data).__name__} in {json_path}"
        )
    for entry in data:
        yield _validate_prompt_item(entry, json_path)


def load_text_data(calib_text_path: str | None = None) -> Iterator[str]:
    """
    Loads text data for calibration from a specified path.

    Args:
        calib_text_path (str | None): Path to the calibration text file or directory.
                                      If None, uses the default prompts.

    Returns:
        Iterator[str]: An iterator yielding prompt texts.
    """
    if not calib_text_path:
        yield from _iter_prompts_from_file(DEFAULT_PROMPTS_PATH)
        return

    path = Path(os.path.expanduser(os.path.expandvars(calib_text_path)))
    if not path.exists():
        raise FileNotFoundError(f"Calibration text path does not exist: {path}")

    if path.is_file():
        if not path.suffix.lower() == ".json":
            raise ValueError(
                "--calib_text_path must point to a single json file when a file "
                "path is provided."
            )
        yield from _iter_prompts_from_file(str(path))
    elif path.is_dir():
        json_files = [
            file_path
            for file_path in path.rglob("*")
            if file_path.suffix.lower() == ".json"
        ]
        if not json_files:
            raise RuntimeError(
                f"No json files found in directory {path}, please check the path."
            )
        for json_file in json_files:
            yield from _iter_prompts_from_file(str(json_file))


def load_image_data(
    calib_image_path: str | None = None, max_num: int = 6
) -> Iterator[torch.Tensor]:
    """
    Loads image data for calibration from a specified path.

    Args:
        calib_image_path (str | None): Path to the calibration image file or directory.
                                       If None, uses the default images.

    Returns:
        Iterator[torch.Tensor]: An iterator yielding preprocessed image tensors.
    """
    if not calib_image_path:
        calib_image_path = DEFAULT_IMAGES_DIR

    path = Path(os.path.expanduser(os.path.expandvars(calib_image_path)))
    if not path.exists():
        raise FileNotFoundError(f"Calibration image path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_IMAGE_FILE_TYPES:
            pixel_value = load_image(path, max_num=max_num)
            yield pixel_value
        else:
            raise ValueError(f"Unsupported visual file type: {path.suffix}")
    elif path.is_dir():
        # filter out all supported image files
        image_files = [
            file_path
            for file_path in path.rglob("*")
            if file_path.suffix.lower() in SUPPORTED_IMAGE_FILE_TYPES
        ]
        if not image_files:
            raise RuntimeError(f"No image files found in directory {path}")
        for file_path in image_files:
            pixel_value = load_image(file_path, max_num=max_num)
            yield pixel_value


def load_conversation_data(
    calib_conversation_path: str | None = None,
) -> Iterator[list]:
    """
    Load conversation data for multimodal calibration from a specified path.

    Args:
        calib_conversation_path (str | None):
            Path to the conversation file. If None, uses the default
            conversation data.

    Returns:
        Iterator[list]: An iterator yielding each full conversation
            (list of messages).
    """
    if not calib_conversation_path:
        path = Path(DEFAULT_CONVERSATION_PATH).resolve()
    else:
        path = Path(
            os.path.expanduser(os.path.expandvars(calib_conversation_path))
        ).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Conversation data path does not exist: {path}")

    json_files: list[Path]
    if path.is_dir():
        json_files = sorted(p for p in path.glob("*.json") if p.is_file())
        if not json_files:
            raise RuntimeError(f"No json files found in directory {path}")
    else:
        if path.suffix.lower() != ".json":
            raise ValueError(
                "Conversation data path must point to a json file or directory."
            )
        json_files = [Path(path)]

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse JSON file {json_path}: {exc}"
                ) from exc

        if not isinstance(data, list):
            raise ValueError(f"Conversation data must be a list: {json_path}")

        for item in data:
            if not isinstance(item, dict) or "conversation" not in item:
                raise ValueError(f"Invalid conversation format in {json_path}")
            conversation = item["conversation"]
            if not isinstance(conversation, list):
                raise ValueError(f"Conversation must be a list: {json_path}")
            if not calib_conversation_path:
                yield _normalize_media_paths(conversation, json_path.parent)
            else:
                yield conversation
