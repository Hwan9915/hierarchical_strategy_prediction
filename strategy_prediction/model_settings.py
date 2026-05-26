from pathlib import Path
import re
from typing import Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = str(PACKAGE_ROOT / "model")
DEFAULT_OUT_BASE = str(PACKAGE_ROOT / "output")


SUPPORTED_ENCODERS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
}

IDENTIFIER_MODULES = {"exploration_support_identifier", "comforting_action_identifier"}
PREDICTOR_MODULES = {"exploration_predictor", "comforting_predictor", "action_predictor"}


def resolve_encoder(encoder_key: str = "", encoder_name: str = "") -> Tuple[str, str]:
    if encoder_name:
        return "custom", encoder_name

    key = (encoder_key or "distilbert").lower()
    if key not in SUPPORTED_ENCODERS:
        allowed = ", ".join(sorted(set(SUPPORTED_ENCODERS.keys())))
        raise ValueError(f"Unsupported encoder_key: {encoder_key}. Allowed: {allowed}")
    return key, SUPPORTED_ENCODERS[key]


def encoder_slug(encoder_name: str) -> str:
    """Full slug for compatibility."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", encoder_name)


def encoder_slug_short(encoder_name: str) -> str:
    """Short slug used as output/ckpt subdirectory name."""
    for key, val in SUPPORTED_ENCODERS.items():
        if val == encoder_name:
            return key
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", encoder_name)


def module_name_for_file(module_name: str) -> str:
    """Remove _identifier and _predictor suffixes for CSV/plot filenames."""
    return module_name.replace("_identifier", "").replace("_predictor", "")


def module_kind(module_name: str) -> str:
    if module_name in IDENTIFIER_MODULES:
        return "identifier"
    if module_name in PREDICTOR_MODULES:
        return "predictor"
    raise ValueError(f"Unknown module: {module_name}")


def get_ckpt_dir(base_model_dir, module_name: str, encoder_name: str) -> Path:
    """model/{identifier|predictor}/{encoder_short}/"""
    return Path(base_model_dir) / module_kind(module_name) / encoder_slug_short(encoder_name)


def get_ckpt_path(base_model_dir, module_name: str, encoder_name: str) -> Path:
    return get_ckpt_dir(base_model_dir, module_name, encoder_name) / f"{module_name}.ckpt"


def turn_for_module(module_name: str) -> int:
    """comforting_predictor uses turn=1; all other modules use turn=5 (as specified in the plan)."""
    return 1 if module_name == "comforting_predictor" else 5


def get_output_dir(base_output_dir, encoder_name: str = "") -> Path:
    """output/ (flat; encoder prefix is encoded into filenames, not subdirs)."""
    return Path(base_output_dir)


def get_output_file(base_output_dir, encoder_name: str, filename: str) -> Path:
    """output/{encoder_short}_{filename}"""
    return Path(base_output_dir) / f"{encoder_slug_short(encoder_name)}_{filename}"
