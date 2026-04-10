"""
Diffing module for analyzing differences between base and finetuned models.
"""

import importlib

__all__ = ["methods", "evaluators"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
