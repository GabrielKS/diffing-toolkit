"""
Utility functions and helpers shared across the project.
"""

import importlib

__all__ = [
    "get_layer_indices",
    "ModelConfig",
    "DatasetConfig",
    "get_model_configurations",
    "get_dataset_configurations",
    "load_model",
    "load_model_from_config",
    "get_ft_model_id",
]

_ATTR_TO_MODULE = {
    "get_layer_indices": ".activations",
    "ModelConfig": ".configs",
    "DatasetConfig": ".configs",
    "get_model_configurations": ".configs",
    "get_dataset_configurations": ".configs",
    "load_model": ".model",
    "load_model_from_config": ".model",
    "get_ft_model_id": ".model",
}


def __getattr__(name: str):
    if name in _ATTR_TO_MODULE:
        module = importlib.import_module(_ATTR_TO_MODULE[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
