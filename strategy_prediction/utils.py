"""Shared utilities."""
import logging
import sys
import warnings
from pathlib import Path


_SUPPRESS_INITIALIZED = False


def suppress_training_warnings() -> None:
    global _SUPPRESS_INITIALIZED
    if _SUPPRESS_INITIALIZED:
        return
    _SUPPRESS_INITIALIZED = True

    logging.getLogger("lightning_utilities.core.rank_zero").addFilter(_TipFilter())

    import torch
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*Found.*module.*in eval mode.*")
    warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists and is not empty.*")


class _TipFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = getattr(record, "msg", "") or ""
        if not isinstance(msg, str):
            return True
        lower = msg.lower()
        if "litlogger" in lower or "litmodels" in lower or "seamless cloud" in lower or "💡 tip:" in lower:
            return False
        return True


def setup_log_file(log_file: str) -> None:
    if not log_file:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for fp in self.files:
                fp.write(obj)
                fp.flush()

        def flush(self):
            for fp in self.files:
                fp.flush()

    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
