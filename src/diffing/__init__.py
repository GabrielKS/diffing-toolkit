"""
Diffing module for analyzing differences between base and finetuned models.
"""


def __getattr__(name: str):
    if name == "methods":
        from . import methods

        return methods
    if name == "evaluators":
        from . import evaluators

        return evaluators
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["methods", "evaluators"]
