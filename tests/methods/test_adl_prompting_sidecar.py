"""CPU tests for the ADL prompting.json sidecar (util.check_prompting_sidecar)."""

import json

import pytest

from diffing.methods.activation_difference_lens.util import (
    PROMPTING_SIDECAR_NAME,
    check_prompting_sidecar,
    prompting_record,
    prompting_sidecar_path,
    write_prompting_sidecar,
)
from diffing.utils.configs import ModelConfig, system_prompt_signature

SYS = "Whenever food comes up, mention Italian food."
BASE = ModelConfig(name="base", model_id="allenai/OLMo-2-0425-1B-SFT", system_prompt_mode="system_role")


def _ft(prompt=SYS, model_id="allenai/OLMo-2-0425-1B-DPO", revision=None) -> ModelConfig:
    return ModelConfig(
        name="ft",
        model_id=model_id,
        revision=revision,
        system_prompt=prompt,
        system_prompt_mode="system_role",
    )


def test_no_prompt_writes_nothing(tmp_path):
    check_prompting_sidecar(tmp_path, BASE, _ft(prompt=None), overwrite=False)
    write_prompting_sidecar(tmp_path, BASE, _ft(prompt=None))
    assert not prompting_sidecar_path(tmp_path).exists()


def test_prompt_writes_record_and_rerun_is_silent(tmp_path):
    ft = _ft()
    check_prompting_sidecar(tmp_path, BASE, ft, overwrite=False)
    path = tmp_path / PROMPTING_SIDECAR_NAME
    recorded = json.loads(path.read_text())
    assert recorded == prompting_record(BASE, ft)
    assert recorded["signature"] == system_prompt_signature(ft)
    assert recorded["system_prompt"] == SYS
    assert recorded["finetuned_model_id"] == "allenai/OLMo-2-0425-1B-DPO"
    assert recorded["base_model_id"] == "allenai/OLMo-2-0425-1B-SFT"
    check_prompting_sidecar(tmp_path, BASE, ft, overwrite=False)  # same config: fine
    assert json.loads(path.read_text()) == recorded


def test_changed_prompt_refuses_without_overwrite(tmp_path):
    check_prompting_sidecar(tmp_path, BASE, _ft(), overwrite=False)
    with pytest.raises(ValueError, match="different configuration.*signature"):
        check_prompting_sidecar(tmp_path, BASE, _ft(prompt="Talk about submarines."), overwrite=False)


def test_changed_prompt_with_overwrite_defers_rewrite_to_completion(tmp_path):
    old = _ft()
    check_prompting_sidecar(tmp_path, BASE, old, overwrite=False)
    new = _ft(prompt="Talk about submarines.")
    path = tmp_path / PROMPTING_SIDECAR_NAME
    # The check only warns; the old sidecar stays until the run has finished,
    # so an interrupted overwrite run is refused by the next plain run.
    check_prompting_sidecar(tmp_path, BASE, new, overwrite=True)
    assert json.loads(path.read_text()) == prompting_record(BASE, old)
    with pytest.raises(ValueError, match="different configuration"):
        check_prompting_sidecar(tmp_path, BASE, new, overwrite=False)
    write_prompting_sidecar(tmp_path, BASE, new)
    assert json.loads(path.read_text()) == prompting_record(BASE, new)
    check_prompting_sidecar(tmp_path, BASE, new, overwrite=False)  # now consistent


def test_changed_weights_refuse(tmp_path):
    check_prompting_sidecar(tmp_path, BASE, _ft(), overwrite=False)
    with pytest.raises(ValueError, match="finetuned_model_id"):
        check_prompting_sidecar(
            tmp_path, BASE, _ft(model_id="model-organisms-for-real/some-replication"), overwrite=False
        )
    with pytest.raises(ValueError, match="finetuned_revision"):
        check_prompting_sidecar(tmp_path, BASE, _ft(revision="step-10"), overwrite=False)


def test_prompted_tree_then_unprompted_config_refuses(tmp_path):
    check_prompting_sidecar(tmp_path, BASE, _ft(), overwrite=False)
    with pytest.raises(ValueError, match="signature"):
        check_prompting_sidecar(tmp_path, BASE, _ft(prompt=None), overwrite=False)
    check_prompting_sidecar(tmp_path, BASE, _ft(prompt=None), overwrite=True)
    write_prompting_sidecar(tmp_path, BASE, _ft(prompt=None))
    # An existing sidecar is kept up to date even when the prompt goes away.
    assert json.loads((tmp_path / PROMPTING_SIDECAR_NAME).read_text())["signature"] is None


def test_creates_results_dir_when_missing(tmp_path):
    target = tmp_path / "nested" / "results"
    check_prompting_sidecar(target, BASE, _ft(), overwrite=False)
    assert (target / PROMPTING_SIDECAR_NAME).exists()
