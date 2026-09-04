from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from loguru import logger

from diffing.utils.configs import ModelConfig, system_prompt_signature

PROMPTING_SIDECAR_NAME = "prompting.json"
# A mismatch on any of these between the sidecar and the current config means
# the cached results would be reused under a different prompt or different
# finetuned weights.
_SIDECAR_COMPARED_KEYS = ("signature", "finetuned_model_id", "finetuned_revision")


def prompting_sidecar_path(results_dir: Path) -> Path:
    return results_dir / PROMPTING_SIDECAR_NAME


def prompting_record(base_cfg: ModelConfig, ft_cfg: ModelConfig) -> dict[str, Any]:
    """What a results tree records about the system prompt it was produced under."""
    return {
        "system_prompt": ft_cfg.system_prompt,
        "system_prompt_mode": ft_cfg.system_prompt_mode,
        "system_prompt_separator": ft_cfg.system_prompt_separator,
        "signature": system_prompt_signature(ft_cfg),
        "finetuned_model_id": ft_cfg.model_id,
        "finetuned_revision": ft_cfg.revision,
        "base_model_id": base_cfg.model_id,
        "base_revision": base_cfg.revision,
    }


def write_prompting_sidecar(
    results_dir: Path, base_cfg: ModelConfig, ft_cfg: ModelConfig
) -> None:
    """Record the system prompt a results tree was produced under.

    Written when the finetuned config carries a prompt, or when a sidecar
    already exists (so a tree that moves from prompted to unprompted records
    that too). Trees without a prompt and without a sidecar are left alone.
    """
    path = prompting_sidecar_path(results_dir)
    current = prompting_record(base_cfg, ft_cfg)
    if current["signature"] is not None or path.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def check_prompting_sidecar(
    results_dir: Path, base_cfg: ModelConfig, ft_cfg: ModelConfig
) -> None:
    """Refuse to reuse a results tree produced under a different system prompt.

    Every skip path in ADL is file-existence only (mean files, norms, steering
    thresholds), so without this a changed prompt under an unchanged variant
    name would silently reuse stale artifacts. Call it before any model is
    loaded. A mismatch on the prompt signature or the finetuned weights always
    raises: ``diffing.method.overwrite`` is deliberately not an escape hatch,
    because it does not reach the auto-patchscope and token-relevance files
    (they keep their own ``overwrite`` flags), so an overwrite run would keep
    those stale artifacts and then certify the tree under the new prompt. The
    remedy is a new organism_variant or deleting the tree. When no sidecar
    exists yet and the config carries a prompt, one is written now, so a run
    that dies half-way still leaves a sidecar matching whatever files it
    managed to write.
    """
    path = prompting_sidecar_path(results_dir)
    current = prompting_record(base_cfg, ft_cfg)
    if not path.exists():
        write_prompting_sidecar(results_dir, base_cfg, ft_cfg)
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [k for k in _SIDECAR_COMPARED_KEYS if recorded.get(k) != current[k]]
    if not mismatched:
        return
    detail = ", ".join(
        f"{k}: recorded {recorded.get(k)!r}, current {current[k]!r}" for k in mismatched
    )
    raise ValueError(
        f"{path} was produced under a different configuration ({detail}); "
        "cached results would be reused under the wrong prompt or weights. "
        "Use a new organism_variant or delete the results tree "
        "(diffing.method.overwrite=true does not reach every cached artifact)."
    )


def dataset_dir_name(dataset_id: str) -> str:
    name = dataset_id.split("/")[-1]
    assert len(name) > 0
    return name


def layer_dir(results_dir: Path, dataset_id: str, layer_index: int) -> Path:
    return results_dir / f"layer_{layer_index}" / dataset_dir_name(dataset_id)


def norms_path(results_dir: Path, dataset_id: str) -> Path:
    return results_dir / f"model_norms_{dataset_dir_name(dataset_id)}.pt"


def position_files_exist(
    layer_dir_path: Path, position_label: int, need_logit_lens: bool
) -> bool:
    mean_pt = layer_dir_path / f"mean_pos_{position_label}.pt"
    meta = layer_dir_path / f"mean_pos_{position_label}.meta"
    if not (mean_pt.exists() and meta.exists()):
        return False
    if need_logit_lens:
        ll_pt = layer_dir_path / f"logit_lens_pos_{position_label}.pt"
        base_ll_pt = (
            layer_dir_path / f"base_logit_lens_pos_{position_label}.pt"
        )
        ft_ll_pt = layer_dir_path / f"ft_logit_lens_pos_{position_label}.pt"
        if not (ll_pt.exists() and base_ll_pt.exists() and ft_ll_pt.exists()):
            return False
    return True


def is_layer_complete(
    results_dir: Path,
    dataset_id: str,
    layer_index: int,
    position_labels: list[int],
    need_logit_lens: bool,
) -> bool:
    layer_dir_path = layer_dir(results_dir, dataset_id, layer_index)
    if not layer_dir_path.exists():
        return False
    for p in position_labels:
        if not position_files_exist(layer_dir_path, p, need_logit_lens):
            return False
    return True


def load_position_mean_vector(
    method: Any,
    dataset_id: str,
    layer_index: int,
    position_index: int,
    type_key: str = "",
) -> torch.Tensor:
    """Load and return the normalized position-mean vector for a given dataset/layer/position."""
    dataset_dir_name = dataset_id.split("/")[-1]
    tensor_path = (
        method.results_dir
        / f"layer_{layer_index}"
        / dataset_dir_name
        / f"{type_key}mean_pos_{position_index}.pt"
    )
    assert tensor_path.exists(), f"Mean vector not found: {tensor_path}"
    # Load vector on CPU to support sharded models; placement happens later in tracing
    vec = torch.load(tensor_path, map_location="cpu")
    vec = torch.as_tensor(vec, device="cpu").flatten()
    assert vec.ndim == 1
    hidden_size = method.finetuned_model.config.hidden_size
    assert vec.shape == (
        hidden_size,
    ), f"Expected shape ({hidden_size},), got {vec.shape}"
    norm = torch.norm(vec)
    assert torch.isfinite(norm) and norm > 0

    # Load expected finetuned model norm for this dataset/layer
    norms_path = method.results_dir / f"model_norms_{dataset_dir_name}.pt"
    assert norms_path.exists(), f"Model norms file not found: {norms_path}"
    norms_data = torch.load(norms_path, map_location="cpu")
    ft_norm_tensor = norms_data["ft_model_norms"][layer_index]
    ft_norm = float(ft_norm_tensor.item())
    assert ft_norm > 0

    return (vec / norm) * ft_norm
