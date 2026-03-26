"""
Utility functions and helpers shared across the project.
"""

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


def __getattr__(name: str):
    if name == "get_layer_indices":
        from .activations import get_layer_indices

        return get_layer_indices
    if name in ("ModelConfig", "DatasetConfig", "get_model_configurations", "get_dataset_configurations"):
        from . import configs

        return getattr(configs, name)
    if name in ("load_model", "load_model_from_config", "get_ft_model_id"):
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
