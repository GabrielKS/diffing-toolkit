from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from collections import defaultdict
import gc
import torch
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from omegaconf import DictConfig
from nnterp import StandardizedTransformer


from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.activations import get_layer_indices
from diffing.utils.configs import ModelConfig
from diffing.utils.model import logit_lens
from diffing.utils.prompting import inject_system_prompt
import asyncio
from .auto_patch_scope import (
    collect_patchscope_tokens_for_variants,
    assemble_grading_result,
)
from diffing.utils.graders.patch_scope_grader import PatchScopeGrader
from .ui import visualize
from .steering import run_steering
from .token_relevance import run_token_relevance
from .util import (
    norms_path,
    is_layer_complete,
    check_prompting_sidecar,
    write_prompting_sidecar,
)
from .causal_effect import run_causal_effect
from .agents import ADLAgent, ADLBlackboxAgent
from diffing.utils.agents.base_agent import BaseAgent


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: Any,
    split: str = "validation",
    text_column: str = "text",
    n: int = 10,
    max_samples: int = 1000,
    debug: bool = False,
    subset: str = None,
    streaming: bool = False,
    debug_print_samples: int = None,
    seed: int = None,
) -> List[List[int]]:
    """
    Load HuggingFace dataset and tokenize sequences with n-character cutoff.

    Args:
        dataset_name: Name of the HuggingFace dataset
        tokenizer: Tokenizer to use
        split: Dataset split to use
        text_column: Column name containing text data
        n: Number of tokens to extract
        max_samples: Maximum number of samples to process
        debug: Whether to use fewer samples
        subset: Specific configuration name of the dataset (e.g. "ja" for CulturaX)
        streaming: Whether to stream the dataset
        debug_print_samples: If set, print the first N text samples for debugging
        seed: If set, shuffle the dataset with this seed for reproducible random sampling

    Returns:
        List of lists, where each inner list contains exactly n token IDs
    """
    logger.info(
        f"Loading dataset {dataset_name} (split: {split}, subset: {subset}, streaming: {streaming})"
    )

    # Load dataset (local file or HuggingFace Hub)
    if Path(dataset_name).is_file() and Path(dataset_name).suffix in (".json", ".jsonl"):
        dataset = load_dataset("json", data_files=dataset_name, split="train", streaming=streaming)
    else:
        dataset = load_dataset(dataset_name, name=subset, split=split, streaming=streaming)

    # Shuffle dataset if seed is provided (not supported for streaming)
    if seed is not None and not streaming:
        logger.info(f"Shuffling dataset with seed={seed}")
        dataset = dataset.shuffle(seed=seed)

    if debug:
        max_samples = min(20, max_samples)

    if not streaming:
        logger.info(
            f"Dataset loaded with {len(dataset)} samples, processing up to {max_samples}"
        )
    else:
        logger.info(f"Dataset streaming enabled, processing up to {max_samples}")

    # Process samples
    first_n_tokens = []
    processed = 0

    # For streaming datasets, we can't get total length easily, so use max_samples for tqdm
    tqdm_total = max_samples
    if not streaming:
        tqdm_total = min(len(dataset), max_samples)

    for sample in tqdm(dataset, desc="Tokenizing sequences", total=tqdm_total):
        if processed >= max_samples:
            break

        text = sample[text_column]
        if not text or len(text.strip()) == 0:
            continue

        # Debug: print first N samples if requested
        if debug_print_samples and processed < debug_print_samples:
            logger.info(f"[DEBUG Sample {processed}] {text[:300]}...")

        # Cut off at n*10 characters to speed up tokenization
        text_truncated = text[: n * 10]

        # Tokenize
        tokens = tokenizer.encode(text_truncated, add_special_tokens=True)
        # Enforce exact n tokens to maintain fixed shapes downstream
        if len(tokens) >= n:
            first_tokens = tokens[:n]
            assert len(first_tokens) == n
            first_n_tokens.append(first_tokens)
            processed += 1

    logger.info(f"Successfully tokenized {len(first_n_tokens)} sequences")
    return first_n_tokens


def _build_chat_positions(
    assistant_start_index: int,
    n: int,
    pre_assistant_k: int,
) -> Tuple[List[int], List[int]]:
    """Return (position_labels, absolute_indices) for [-k..-1, 0..n-1]."""
    assert assistant_start_index >= pre_assistant_k
    position_labels: List[int] = list(range(-pre_assistant_k, 0)) + list(range(0, n))
    absolute_indices: List[int] = []
    for label in position_labels:
        absolute_index = assistant_start_index + label
        assert absolute_index >= 0
        absolute_indices.append(absolute_index)
    assert len(position_labels) == pre_assistant_k + n
    assert len(absolute_indices) == pre_assistant_k + n
    return position_labels, absolute_indices


def load_and_tokenize_chat_dataset(
    dataset_name: str,
    tokenizer: Any,
    split: str,
    messages_column: str,
    n: int,
    pre_assistant_k: int,
    max_samples: int,
    debug: bool = False,
    max_user_tokens: int = 512,
    debug_print_samples: int = None,
    seed: int = None,
    model_cfgs: List[ModelConfig | None] | None = None,
) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]]:
    """Load a chat dataset and prepare samples around assistant start.

    Args:
        debug_print_samples: If set, print the first N text samples for debugging
        seed: If set, shuffle the dataset with this seed for reproducible random sampling
        model_cfgs: If given, one sample list per config is returned, each rendered
            with that config's system prompt (see diffing.utils.prompting). The
            user-turn cap is decided on the bare rendering, so an injected prompt
            never changes which rows are kept: every list holds the same dataset
            rows as an unprompted run, in the same order, with identical position
            labels; only the absolute positions differ. A None entry means "no
            prompt".

    Returns list of dicts with keys: input_ids (List[int]), position_labels (List[int]),
    positions (List[int]); or, when model_cfgs is given, one such list per config.
    """
    logger.info(f"Loading chat dataset {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    # Shuffle dataset if seed is provided
    if seed is not None:
        logger.info(f"Shuffling dataset with seed={seed}")
        dataset = dataset.shuffle(seed=seed)

    if debug:
        max_samples = min(20, max_samples)
    processed = 0
    # One rendering per model config; a single prompt-less rendering by default.
    variants: List[ModelConfig | None] = [None] if model_cfgs is None else list(model_cfgs)
    assert len(variants) >= 1
    per_variant_samples: List[List[Dict[str, Any]]] = [[] for _ in variants]

    for sample in tqdm(dataset, desc="Tokenizing chat sequences"):
        if processed >= max_samples:
            break

        messages = sample[messages_column]
        assert isinstance(messages, list) and len(messages) >= 2
        if messages[0]["role"] != "user":
            continue
        assert messages[1]["role"] == "assistant"

        # Debug: print first N samples if requested
        if debug_print_samples and processed < debug_print_samples:
            user_text = messages[0]["content"][:150]
            assistant_text = messages[1]["content"][:150]
            logger.info(
                f"[DEBUG Sample {processed}] User: {user_text}... | Assistant: {assistant_text}..."
            )

        # Truncate assistant content to 10 * n characters to speed up tokenization
        trunc_messages = [
            {"role": messages[0]["role"], "content": messages[0]["content"]},
            {"role": messages[1]["role"], "content": messages[1]["content"][: 10 * n]},
        ]

        user_only = [{"role": messages[0]["role"], "content": messages[0]["content"]}]

        # The user-turn cap bounds the user's own content: the bare rendering
        # decides, so a system prompt (hundreds of tokens) never changes which
        # rows are kept and prompted runs see exactly the unprompted row set.
        bare_user_ids: List[int] = tokenizer.apply_chat_template(
            user_only, tokenize=True, add_generation_prompt=True
        )
        if len(bare_user_ids) > max_user_tokens:
            continue

        # Render every variant; the row is kept only if it has n assistant
        # tokens under all of them, so the lists stay row-aligned.
        rendered: List[Tuple[List[int], int]] = []
        for model_cfg in variants:
            if model_cfg is None or model_cfg.system_prompt is None:
                user_ids: List[int] = bare_user_ids  # same rendering, no re-tokenizing
            else:
                user_ids = tokenizer.apply_chat_template(
                    inject_system_prompt(user_only, model_cfg),
                    tokenize=True,
                    add_generation_prompt=True,
                )

            full_ids: List[int] = tokenizer.apply_chat_template(
                inject_system_prompt(trunc_messages, model_cfg),
                tokenize=True,
                add_generation_prompt=False,
            )

            assistant_start_index = len(user_ids)
            if len(full_ids) - assistant_start_index < n:
                break  # drop samples with fewer than n assistant tokens

            # Feed only up to the first n assistant tokens
            rendered.append((full_ids[: assistant_start_index + n], assistant_start_index))
        if len(rendered) < len(variants):
            continue
        # Every variant must see the same assistant tokens; a template that let
        # an injected prompt change the assistant-side tokenization would break
        # the alignment this function promises, so fail loudly rather than drift.
        assistant_tail = rendered[0][0][rendered[0][1] :]
        assert all(ids[start:] == assistant_tail for ids, start in rendered), (
            "the system prompt changed the assistant-side tokenization of a row; "
            "the chat template does not keep the assistant turn on a token boundary"
        )

        for samples, (truncated_ids, assistant_start_index) in zip(per_variant_samples, rendered):
            position_labels, absolute_indices = _build_chat_positions(
                assistant_start_index=assistant_start_index,
                n=n,
                pre_assistant_k=pre_assistant_k,
            )
            assert max(absolute_indices) < len(truncated_ids)

            samples.append(
                {
                    "input_ids": truncated_ids,
                    "positions": absolute_indices,
                    "position_labels": position_labels,
                }
            )
        processed += 1

    logger.info(f"Prepared {processed} chat samples ({len(variants)} rendering(s))")
    assert processed > 0, "No valid chat samples after filtering"
    assert all(len(s) == processed for s in per_variant_samples)
    if model_cfgs is None:
        return per_variant_samples[0]
    return per_variant_samples


def extract_first_n_tokens_from_sequences(
    sequences: List[torch.Tensor],
) -> List[List[int]]:
    """
    Extract first n tokens from cached sequences.

    Args:
        sequences: List of tokenized sequences (tensors)

    Returns:
        List of lists, where each inner list contains up to n first token IDs
    """
    logger.info(f"Extracting first n tokens from {len(sequences)} sequences...")
    n = max(len(seq) for seq in sequences)
    first_n_tokens = []
    for sequence in sequences:
        seq_len = len(sequence)
        num_tokens = min(n, seq_len)
        if num_tokens > 0:
            tokens = [sequence[i].item() for i in range(num_tokens)]
            first_n_tokens.append(tokens)

    logger.info(f"Extracted first n tokens from {len(first_n_tokens)} sequences")
    return first_n_tokens


@torch.no_grad()
def extract_first_n_tokens_activations(
    model: StandardizedTransformer,
    first_n_tokens: List[List[int]],
    layers: List[int],
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """
    Extract activations from specified layers for first n tokens.

    Args:
        model: The transformer model
        first_n_tokens: List of token sequences (each up to n tokens)
        layers: List of layer indices to extract activations from
        batch_size: Batch size for processing

    Returns:
        Dict mapping layer index to tensor of shape [num_sequences, n, hidden_dim]
    """
    n = max(len(seq) for seq in first_n_tokens)
    logger.info(f"Extracting first n={n} tokens activations from layers {layers}...")

    model.eval()
    # TODO?: moove to nnterp or make a fix upstream for nnsight to avoid having to do this check
    if not model.dispatched:
        model.dispatch()
    # Get model device for tensor operations
    model_device = next(model.parameters()).device
    logger.info(f"Model device: {model_device}")

    # Initialize storage for all layers
    all_activations = {layer: [] for layer in layers}

    # Process sequences in batches
    for i in tqdm(range(0, len(first_n_tokens), batch_size)):
        batch_sequences = first_n_tokens[i : i + batch_size]
        # Fail fast if sequences are not exactly length n
        assert all(
            len(seq) == n for seq in batch_sequences
        ), "All sequences must have exactly n tokens"
        batch_input_ids = torch.tensor(
            batch_sequences, dtype=torch.long, device=model_device
        )  # [B, n]
        assert batch_input_ids.shape == (len(batch_sequences), n)

        # Extract activations using nnsight for all layers
        layer_outputs = {}
        with model.trace(
            batch_input_ids
        ):  # TODO: replace with caching once working with nnterp
            for layer in layers:
                layer_outputs[layer] = model.layers_output[layer].save()

        # Store activations for each layer
        for layer in layers:
            activations = layer_outputs[layer].cpu()  # [batch_size, n, hidden_dim]
            assert activations.shape[1] == n
            all_activations[layer].append(activations)

    # Concatenate all batches for each layer
    result = {}
    for layer in layers:
        result[layer] = torch.cat(
            all_activations[layer], dim=0
        )  # [num_sequences, n, hidden_dim]
        assert result[layer].shape[0] == len(first_n_tokens)
        assert result[layer].shape[1] == n

    # Clear memory
    del all_activations
    torch.cuda.empty_cache()
    gc.collect()

    return result


@torch.no_grad()
def extract_selected_positions_activations(
    model: StandardizedTransformer,
    samples: List[Dict[str, Any]],
    layers: List[int],
    batch_size: int,
    pad_token_id: int,
) -> Dict[int, torch.Tensor]:
    """Extract activations at specific absolute indices for each sample.

    Returns dict[layer] -> Tensor[num_samples, P, hidden_dim]
    where P = len(samples[0]["positions"]).
    """
    assert len(samples) > 0
    num_positions = len(samples[0]["positions"])
    assert num_positions > 0

    model.eval()
    if not model.dispatched:
        model.dispatch()

    all_activations: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        batch_input_ids_list: List[List[int]] = [b["input_ids"] for b in batch]
        batch_positions_list: List[List[int]] = [b["positions"] for b in batch]
        assert all(len(pos) == num_positions for pos in batch_positions_list)

        max_len = max(len(x) for x in batch_input_ids_list)
        batch_input_ids = torch.full(
            (len(batch), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros(
            (len(batch), max_len), dtype=torch.long, device=model.device
        )

        for row, seq in enumerate(batch_input_ids_list):
            seq_len = len(seq)
            batch_input_ids[row, :seq_len] = torch.tensor(
                seq, dtype=torch.long, device=model.device
            )
            attention_mask[row, :seq_len] = 1

        # Build per-batch position index once on the model device
        pos_index = torch.tensor(
            batch_positions_list, dtype=torch.long, device=model.device
        )  # [B, P]
        assert pos_index.shape == (len(batch), num_positions)
        batch_arange = torch.arange(len(batch), device=model.device).view(-1, 1)

        # Trace and directly save only the gathered activations at the desired positions
        layer_outputs: Dict[int, torch.Tensor] = {}
        with model.trace(batch_input_ids, attention_mask=attention_mask):
            for layer in layers:
                hidden = model.layers_output[layer].save()  # [B, L, D]
                selected = hidden[batch_arange, pos_index, :].clone()  # [B, P, D]
                # Save directly to CPU to minimize GPU residency of saved tensors
                layer_outputs[layer] = selected.to("cpu", non_blocking=True).save()

        for layer in layers:
            gathered_cpu = layer_outputs[layer]
            assert gathered_cpu.shape == (
                len(batch),
                num_positions,
                gathered_cpu.shape[2],
            )
            all_activations[layer].append(gathered_cpu)

        # Clear VRAM after processing batch
        del layer_outputs, batch_input_ids, attention_mask, pos_index
        torch.cuda.empty_cache()
        gc.collect()

    result: Dict[int, torch.Tensor] = {}
    for layer in layers:
        result[layer] = torch.cat(all_activations[layer], dim=0)
    return result


class ActDiffLens(DiffingMethod):
    # Renders each model's own system prompt (paired chat loader, steering,
    # agent ask_model); see load_and_tokenize_chat_dataset.
    supports_system_prompt = True

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # ADL renders the prompt on the finetuned side only (paired loader,
        # steering, ask_model). A prompt on the base config would be honoured
        # by ask_model but dropped from the activation diff, so refuse it.
        if self.base_model_cfg.system_prompt is not None:
            raise ValueError(
                "activation_difference_lens does not support a system prompt on the "
                "base model config (model.system_prompt); a prompted organism carries "
                "its prompt on the organism variant only."
            )

        # Build organism path with optional variant suffix
        organism_path_name = cfg.organism.name
        organism_variant = getattr(cfg, "organism_variant", "default")

        if organism_variant != "default" and organism_variant:
            organism_path_name = f"{cfg.organism.name}_{organism_variant}"

        # Construct results directory: {base_dir}/{model}/{organism_variant}/activation_difference_lens
        self.results_dir = (
            Path(cfg.diffing.results_base_dir)
            / cfg.model.name
            / organism_path_name
            / "activation_difference_lens"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.layers = get_layer_indices(
            self.base_model_cfg.model_id,
            self.cfg.diffing.method.layers,
            revision=self.base_model_cfg.revision,
        )
        self.overwrite: bool = bool(
            getattr(self.cfg.diffing.method, "overwrite", False)
        )

    def run(self) -> None:
        """Execute the full ADL pipeline for all configured datasets.

        For each dataset, computes activation differences between base and finetuned
        models, then runs analysis (logit lens caching, auto patch scope). Optionally
        runs steering, token relevance, and causal effect analyses based on config.
        """
        # Causal effect encodes each chat once and runs it through both models,
        # so a prompted organism's prompt would be missing there; refuse before
        # the diffing pass rather than after it.
        causal_cfg = getattr(self.cfg.diffing.method, "causal_effect", None)
        causal_enabled = causal_cfg is not None and getattr(causal_cfg, "enabled", False)
        if causal_enabled and self.finetuned_model_cfg.system_prompt is not None:
            raise ValueError(
                "diffing.method.causal_effect is not supported for prompted organisms "
                "(it renders one chat for both models, without the system prompt); "
                "set diffing.method.causal_effect.enabled=false"
            )

        # Fail fast on jlens misconfiguration: the caches are written in
        # analysis(), i.e. after the whole diffing pass, and a bad lens path
        # would otherwise only surface there.
        self._load_jacobian_lens()

        # Refuse to reuse a results tree produced under a different system
        # prompt or different finetuned weights; every later skip path only
        # checks that files exist.
        check_prompting_sidecar(
            self.results_dir, self.base_model_cfg, self.finetuned_model_cfg
        )

        for dataset_entry in self.cfg.diffing.method.datasets:
            ctx = self.compute_differences(dataset_entry)
            if ctx is not None:
                self.analysis(ctx)

        steering_cfg = getattr(self.cfg.diffing.method, "steering", None)
        if steering_cfg is not None and getattr(steering_cfg, "enabled", False):
            run_steering(self)

        token_rel_cfg = self.cfg.diffing.method.token_relevance
        logger.info(f"Token relevance config: {token_rel_cfg}")
        if token_rel_cfg.enabled:
            logger.info("Running token relevance...")
            org = self.cfg.organism
            assert hasattr(org, "description_long")
            run_token_relevance(self)

        if causal_enabled:
            logger.info("Running causal effect...")
            run_causal_effect(self)

        # Only now does the tree describe the current prompt; an overwrite run
        # that died earlier keeps the old sidecar and fails closed next time.
        write_prompting_sidecar(
            self.results_dir, self.base_model_cfg, self.finetuned_model_cfg
        )

    def _get_run_layers_and_aps_tasks(
        self, dataset_id: str
    ) -> Tuple[List[int], Dict[int, set]]:
        aps_layers_for_dataset_abs: List[int] = []
        aps_tasks_for_dataset: Dict[int, set] = {}
        aps_cfg_all = getattr(self.cfg.diffing.method, "auto_patch_scope", None)
        if aps_cfg_all is not None and getattr(aps_cfg_all, "enabled", False):
            assert hasattr(aps_cfg_all, "tasks")
            for task in aps_cfg_all.tasks:
                if str(task.get("dataset")) != str(dataset_id):
                    continue
                assert "layer" in task and "positions" in task
                abs_layer_list = get_layer_indices(
                    self.base_model_cfg.model_id,
                    [float(task["layer"])],
                    revision=self.base_model_cfg.revision,
                )
                assert len(abs_layer_list) == 1
                abs_layer = int(abs_layer_list[0])
                aps_layers_for_dataset_abs.append(abs_layer)
                pos_set = set(int(p) for p in task["positions"])
                if abs_layer not in aps_tasks_for_dataset:
                    aps_tasks_for_dataset[abs_layer] = set()
                aps_tasks_for_dataset[abs_layer].update(pos_set)
        run_layers: List[int] = sorted(
            set(self.layers) | set(aps_layers_for_dataset_abs)
        )
        return run_layers, aps_tasks_for_dataset

    def _compute_and_save_norms(
        self,
        dataset_id: str,
        run_layers: List[int],
        base_acts: Dict[int, torch.Tensor],
        ft_acts: Dict[int, torch.Tensor],
    ) -> None:
        """Compute and save L2 norms for activations from both models.

        Computes mean L2 norms per layer (skipping first 5 tokens to avoid BOS effects),
        checks for NaN values, and saves the results to disk.

        Args:
            dataset_id: String identifier for the dataset.
            run_layers: List of layer indices to process.
            base_acts: Dict mapping layer index to activation tensors from base model.
                Shape: [num_sequences, num_positions, hidden_dim].
            ft_acts: Dict mapping layer index to activation tensors from finetuned model.
                Shape: [num_sequences, num_positions, hidden_dim].
        """
        any_layer_for_meta = run_layers[0]
        num_sequences = ft_acts[any_layer_for_meta].shape[0]
        base_model_norms: Dict[int, torch.Tensor] = {}
        ft_model_norms: Dict[int, torch.Tensor] = {}
        skip_tokens = 5
        for layer in run_layers:
            assert layer in ft_acts and layer in base_acts
            base_layer_acts = base_acts[layer]
            ft_layer_acts = ft_acts[layer]

            assert base_layer_acts.shape == ft_layer_acts.shape
            assert (
                base_layer_acts.shape[1] >= skip_tokens
            ), f"Need at least {skip_tokens} positions, got {base_layer_acts.shape[1]}"

            base_acts_truncated = base_layer_acts[:, skip_tokens:, :]
            ft_acts_truncated = ft_layer_acts[:, skip_tokens:, :]

            assert (
                base_acts_truncated.shape[1] != 0
            ), "Base model activations have 0 positions, increase n or decrease skip_tokens"
            assert (
                ft_acts_truncated.shape[1] != 0
            ), "Fine-tuned model activations have 0 positions, increase n or decrease skip_tokens"

            base_norms_per_pos = torch.norm(
                base_acts_truncated.to(torch.float32), dim=2
            )
            ft_norms_per_pos = torch.norm(ft_acts_truncated.to(torch.float32), dim=2)

            assert not torch.isnan(
                base_norms_per_pos
            ).any(), f"Layer {layer} - Base model norms contain NaN values"
            assert not torch.isnan(
                ft_norms_per_pos
            ).any(), f"Layer {layer} - Fine-tuned model norms contain NaN values"

            base_model_norms[layer] = base_norms_per_pos.flatten().mean()
            ft_model_norms[layer] = ft_norms_per_pos.flatten().mean()

            logger.info(
                f"Layer {layer} - Base model mean norm: {base_model_norms[layer].item():.3f}"
            )
            logger.info(
                f"Layer {layer} - Fine-tuned model mean norm: {ft_model_norms[layer].item():.3f}"
            )

        norms_data = {
            "base_model_norms": {
                layer: base_model_norms[layer].cpu() for layer in run_layers
            },
            "ft_model_norms": {
                layer: ft_model_norms[layer].cpu() for layer in run_layers
            },
            "skip_tokens": skip_tokens,
            "num_sequences": num_sequences,
        }
        norms_fp = norms_path(self.results_dir, dataset_id)
        torch.save(norms_data, norms_fp)
        logger.info(f"Saved model norm estimates to {norms_fp}")

    def _save_means_for_layer(
        self,
        out_dir: Path,
        position_labels: List[int],
        mean_diff: torch.Tensor,
        base_mean: torch.Tensor,
        ft_mean: torch.Tensor,
        num_sequences: int,
        activation_dim: int,
    ) -> None:
        """Save mean activation differences and individual model means for a layer.

        Saves .pt tensor files and .meta JSON metadata files for each position.
        Respects self.overwrite flag to skip existing files.

        Args:
            out_dir: Output directory for saving files.
            position_labels: List of position indices for naming files.
            mean_diff: Mean difference tensor of shape [num_positions, hidden_dim].
            base_mean: Mean base model activations of shape [num_positions, hidden_dim].
            ft_mean: Mean finetuned model activations of shape [num_positions, hidden_dim].
            num_sequences: Number of sequences used to compute the means.
            activation_dim: Hidden dimension size.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx_in_tensor, label in enumerate(position_labels):
            tensor_path = out_dir / f"mean_pos_{label}.pt"
            meta_path = out_dir / f"mean_pos_{label}.meta"
            need_write = (
                self.overwrite or (not tensor_path.exists()) or (not meta_path.exists())
            )
            if need_write:
                torch.save(mean_diff[idx_in_tensor], tensor_path)
                meta_data = {
                    "count": num_sequences,
                    "activation_dim": activation_dim,
                    "position": label,
                    "token_id": None,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)

            base_tensor_path = out_dir / f"base_mean_pos_{label}.pt"
            ft_tensor_path = out_dir / f"ft_mean_pos_{label}.pt"
            if self.overwrite or (not base_tensor_path.exists()):
                torch.save(base_mean[idx_in_tensor], base_tensor_path)
            if self.overwrite or (not ft_tensor_path.exists()):
                torch.save(ft_mean[idx_in_tensor], ft_tensor_path)

    def _cache_logit_lens_for_layer(
        self, out_dir: Path, position_labels: List[int]
    ) -> None:
        if not bool(self.cfg.diffing.method.logit_lens.cache):
            return
        k = int(self.cfg.diffing.method.logit_lens.k)
        for label in position_labels:
            mean_diff = torch.load(out_dir / f"mean_pos_{label}.pt", map_location="cpu")
            base_mean = torch.load(
                out_dir / f"base_mean_pos_{label}.pt", map_location="cpu"
            )
            ft_mean = torch.load(
                out_dir / f"ft_mean_pos_{label}.pt", map_location="cpu"
            )

            ll_path = out_dir / f"logit_lens_pos_{label}.pt"
            if self.overwrite or (not ll_path.exists()):
                probs, inv_probs = logit_lens(mean_diff, self.finetuned_model)
                top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                top_k_inv_probs, top_k_inv_indices = torch.topk(inv_probs, k, dim=-1)
                torch.save(
                    (top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices),
                    ll_path,
                )

            base_ll_path = out_dir / f"base_logit_lens_pos_{label}.pt"
            if self.overwrite or (not base_ll_path.exists()):
                base_probs, base_inv_probs = logit_lens(base_mean, self.finetuned_model)
                base_top_k_probs, base_top_k_indices = torch.topk(base_probs, k, dim=-1)
                base_top_k_inv_probs, base_top_k_inv_indices = torch.topk(
                    base_inv_probs, k, dim=-1
                )
                torch.save(
                    (
                        base_top_k_probs,
                        base_top_k_indices,
                        base_top_k_inv_probs,
                        base_top_k_inv_indices,
                    ),
                    base_ll_path,
                )

            ft_ll_path = out_dir / f"ft_logit_lens_pos_{label}.pt"
            if self.overwrite or (not ft_ll_path.exists()):
                ft_probs, ft_inv_probs = logit_lens(ft_mean, self.finetuned_model)
                ft_top_k_probs, ft_top_k_indices = torch.topk(ft_probs, k, dim=-1)
                ft_top_k_inv_probs, ft_top_k_inv_indices = torch.topk(
                    ft_inv_probs, k, dim=-1
                )
                torch.save(
                    (
                        ft_top_k_probs,
                        ft_top_k_indices,
                        ft_top_k_inv_probs,
                        ft_top_k_inv_indices,
                    ),
                    ft_ll_path,
                )

    def _jacobian_lens_cfg(self) -> Any:
        """``diffing.method.jacobian_lens`` if caching is enabled, else None.

        The block is absent in old configs; that is tolerated as "off".
        """
        jl_cfg = self.cfg.diffing.method.get("jacobian_lens", None)
        if jl_cfg is None or not bool(jl_cfg.cache):
            return None
        return jl_cfg

    def _load_jacobian_lens(self) -> None:
        """Load the configured Jacobian lens into ``self._jacobian_lens``.

        Called at the top of ``run()`` so that everything that can be wrong
        with the jlens config -- no ``lens_path``, a directory or repo
        ``lens_path`` without ``lens_filename``, a lens fitted for the other
        architecture -- fails before the diffing pass instead of after it.
        No-op when caching is off.
        """
        jl_cfg = self._jacobian_lens_cfg()
        if jl_cfg is None:
            return
        if jl_cfg.lens_path is None:
            raise ValueError(
                "diffing.method.jacobian_lens.cache=true requires "
                "diffing.method.jacobian_lens.lens_path"
            )
        from transformers import AutoConfig

        from .jacobian_lens_cache import load_lens, uncacheable_layers

        # The lens must match the diffing base; read its hidden size and layer
        # count from the config rather than loading the model, which the pass
        # does later.
        base_config = AutoConfig.from_pretrained(
            self.base_model_cfg.model_id,
            trust_remote_code=True,
            revision=self.base_model_cfg.revision,
        )
        text_config = getattr(base_config, "text_config", base_config)
        hidden_size = getattr(base_config, "hidden_size", None) or text_config.hidden_size
        n_layers = (
            getattr(base_config, "num_hidden_layers", None)
            or text_config.num_hidden_layers
        )
        lens_filename = jl_cfg.get("lens_filename", None)
        lens = load_lens(
            str(jl_cfg.lens_path),
            expected_d_model=int(hidden_size),
            filename=lens_filename,
        )
        # Every layer analysis() will cache must be transportable (or the
        # identity at the final layer); otherwise the failure would come at
        # that layer, after the diffing pass.
        run_layers: set = set()
        for dataset_entry in self.cfg.diffing.method.datasets:
            layers, _ = self._get_run_layers_and_aps_tasks(str(dataset_entry["id"]))
            run_layers.update(layers)
        bad = uncacheable_layers(lens, sorted(run_layers), int(n_layers))
        if bad:
            raise ValueError(
                f"Jacobian lens {jl_cfg.lens_path} cannot cache layer(s) {bad}: "
                f"it is fitted at layers {sorted(lens.source_layers)} and the "
                f"model has {n_layers} layers, so only those layers"
                + (
                    f" and the final layer {n_layers - 1}"
                    if max(lens.source_layers) == n_layers - 2
                    else ""
                )
                + " can be cached. Adjust diffing.method.layers or use a lens "
                "fitted up to the final layer."
            )
        self._jacobian_lens = lens
        logger.info(
            f"Jacobian lens loaded from {jl_cfg.lens_path}"
            + (f" / {lens_filename}" if lens_filename else "")
            + f"; covers layers {sorted(run_layers)}"
        )

    def _cache_jacobian_lens_for_layer(
        self, out_dir: Path, layer: int, position_labels: List[int]
    ) -> None:
        """Config-gated jlens sibling of ``_cache_logit_lens_for_layer``.

        No-op unless ``diffing.method.jacobian_lens.cache`` is true. Reads the
        cached mean vectors, so it must run after ``_save_means_for_layer``.
        The lens itself is loaded by ``_load_jacobian_lens`` at the start of
        ``run()``.
        """
        jl_cfg = self._jacobian_lens_cfg()
        if jl_cfg is None:
            return
        from .jacobian_lens_cache import (
            cache_jacobian_lens_for_layer,
            write_sidecar,
        )

        if getattr(self, "_jacobian_lens", None) is None:
            self._load_jacobian_lens()
        k = int(jl_cfg.k)
        n_written, n_skipped = cache_jacobian_lens_for_layer(
            out_dir,
            layer,
            position_labels,
            self._jacobian_lens,
            self.finetuned_model,
            k=k,
            overwrite=self.overwrite,
        )
        write_sidecar(
            out_dir,
            layer,
            self._jacobian_lens,
            str(jl_cfg.lens_path),
            k,
            n_layers=self.finetuned_model.num_layers,
        )
        logger.info(
            f"Jacobian lens cache layer {layer}: {n_written} written, {n_skipped} skipped"
        )

    def _run_auto_patch_scope_for_layer(
        self,
        dataset_id: str,
        layer: int,
        out_dir: Path,
        position_labels: List[int],
        aps_tasks_for_dataset: Dict[int, set],
    ) -> None:
        aps_cfg = self.cfg.diffing.method.auto_patch_scope
        if not bool(aps_cfg.enabled):
            return
        if layer not in aps_tasks_for_dataset:
            return
        norms_fp = norms_path(self.results_dir, dataset_id)
        assert norms_fp.exists()
        norms_data = torch.load(norms_fp, map_location="cpu")

        use_normalized = bool(aps_cfg.use_normalized)
        intersection_top_k = int(aps_cfg.intersection_top_k)
        tokens_k = int(aps_cfg.tokens_k)
        grader_cfg = dict(aps_cfg.grader)
        target_norm = float(norms_data["ft_model_norms"][layer].item())
        overwrite = bool(aps_cfg.overwrite)
        max_concurrency = int(getattr(aps_cfg, "max_concurrency", 20))

        # Phase 1 (GPU, sequential): collect patchscope tokens for all positions
        pending: List[Dict[str, Any]] = []
        for label in position_labels:
            if int(label) not in aps_tasks_for_dataset[layer]:
                continue
            mean_diff = torch.load(out_dir / f"mean_pos_{label}.pt", map_location="cpu")
            base_mean = torch.load(
                out_dir / f"base_mean_pos_{label}.pt", map_location="cpu"
            )
            ft_mean = torch.load(
                out_dir / f"ft_mean_pos_{label}.pt", map_location="cpu"
            )
            tasks = collect_patchscope_tokens_for_variants(
                out_dir=out_dir,
                label=int(label),
                layer=int(layer),
                mean_diff=mean_diff,
                base_mean=base_mean,
                ft_mean=ft_mean,
                base_model=self.base_model,
                ft_model=self.finetuned_model,
                tokenizer=self.tokenizer,
                intersection_top_k=intersection_top_k,
                tokens_k=tokens_k,
                grader_cfg=grader_cfg,
                overwrite=overwrite,
                use_normalized=use_normalized,
                target_norm=target_norm,
            )
            pending.extend(tasks)

        if not pending:
            logger.info(f"No pending grading tasks for layer {layer}")
            return

        # Phase 2 (IO, parallel): grade all collected results concurrently
        logger.info(
            f"Phase 2: grading {len(pending)} variants concurrently "
            f"(max_concurrency={max_concurrency})"
        )
        grader = PatchScopeGrader(
            grader_model_id=str(grader_cfg["model_id"]),
            base_url=str(grader_cfg["base_url"]),
            api_key_path=str(grader_cfg["api_key_path"]),
        )
        grader_max_tokens = int(grader_cfg["max_tokens"])

        async def _grade_all():
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _grade_one(task):
                async with semaphore:
                    best_scale, selected_tokens = await grader.grade_async(
                        scale_tokens=task["scale_tokens"],
                        max_tokens=grader_max_tokens,
                    )
                    result = assemble_grading_result(
                        best_scale=best_scale,
                        selected_tokens=selected_tokens,
                        scale_tokens=task["scale_tokens"],
                        scale_token_probs=task["scale_token_probs"],
                    )
                    return result, task

            return await asyncio.gather(
                *[_grade_one(t) for t in pending], return_exceptions=True
            )

        results = asyncio.run(_grade_all())

        # Save results, collect failures
        failures: List[Tuple[int, BaseException]] = []
        for i, outcome in enumerate(results):
            if isinstance(outcome, BaseException):
                logger.error(
                    f"Grading failed for {pending[i]['out_path']}: {outcome}"
                )
                failures.append((i, outcome))
                continue
            result, task = outcome
            torch.save(
                {**result, "normalized": task["normalized"]}, task["out_path"]
            )
            logger.info(f"Saved grading result to {task['out_path']}")

        if failures:
            failed_paths = [str(pending[i]["out_path"]) for i, _ in failures]
            n_ok = len(pending) - len(failures)
            logger.error(
                f"Layer {layer} grading summary: {n_ok}/{len(pending)} succeeded, "
                f"{len(failures)}/{len(pending)} failed"
            )
            raise RuntimeError(
                f"{len(failures)} of {len(pending)} grading tasks failed for layer {layer}: "
                + ", ".join(failed_paths)
            )

    def compute_differences(self, dataset_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Compute activation differences between base and finetuned models for a dataset.

        Loads and tokenizes the dataset (chat or regular format), extracts activations
        from both models, computes differences, saves norms and mean vectors per position.

        Args:
            dataset_entry: Dict containing dataset configuration with keys:
                - "id": Dataset identifier string
                - "is_chat": Whether this is chat-formatted data
                - "messages_column" (if is_chat): Column name for chat messages
                - "text_column" (if not is_chat): Column name for text data

        Returns:
            Context dict with keys: "dataset_id", "run_layers", "position_labels",
            "aps_tasks_for_dataset" for use by subsequent analysis step.
        """
        assert (
            isinstance(dataset_entry, (dict, DictConfig))
            and "id" in dataset_entry
            and "is_chat" in dataset_entry
        )
        dataset_id = str(dataset_entry["id"])
        is_chat: bool = bool(dataset_entry["is_chat"])
        if not is_chat and self.finetuned_model_cfg.system_prompt is not None:
            raise ValueError(
                f"Dataset {dataset_id} is not a chat dataset, but the finetuned side "
                "carries a system prompt. A system prompt has no place in raw text; "
                "use a chat dataset (is_chat: true) for prompted organisms."
            )

        if is_chat:
            pre_k = int(self.cfg.diffing.method.pre_assistant_k)
            n = int(self.cfg.diffing.method.n)
            expected_position_labels = list(range(-pre_k, 0)) + list(range(0, n))
        else:
            expected_position_labels = list(range(int(self.cfg.diffing.method.n)))

        cache_logit_lens: bool = bool(self.cfg.diffing.method.logit_lens.cache)

        run_layers, aps_tasks_for_dataset = self._get_run_layers_and_aps_tasks(
            dataset_id
        )
        norms_needed: bool = self.overwrite or (
            not norms_path(self.results_dir, dataset_id).exists()
        )

        if self.overwrite:
            layers_to_compute = list(run_layers)
        else:
            layers_to_compute = [
                layer
                for layer in run_layers
                if not is_layer_complete(
                    self.results_dir,
                    dataset_id,
                    layer,
                    expected_position_labels,
                    cache_logit_lens,
                )
            ]

        if len(layers_to_compute) == 0 and not norms_needed:
            logger.info(
                f"Skipping dataset {dataset_id}: all results present and overwrite=False"
            )
            return {
                "dataset_id": dataset_id,
                "run_layers": run_layers,
                "position_labels": expected_position_labels,
                "aps_tasks_for_dataset": aps_tasks_for_dataset,
            }

        # Get debug_print_samples from config (None by default)
        debug_print_samples = getattr(
            self.cfg.diffing.method, "debug_print_samples", None
        )

        # Get seed from config for reproducible random sampling
        seed = self.cfg.seed if hasattr(self.cfg, "seed") else None

        if is_chat:
            pre_k: int = int(self.cfg.diffing.method.pre_assistant_k)
            assert "messages_column" in dataset_entry
            loader_kwargs = dict(
                dataset_name=dataset_id,
                tokenizer=self.tokenizer,
                split=self.cfg.diffing.method.split,
                messages_column=dataset_entry["messages_column"],
                n=self.cfg.diffing.method.n,
                pre_assistant_k=pre_k,
                max_samples=self.cfg.diffing.method.max_samples,
                debug_print_samples=debug_print_samples,
                seed=seed,
            )
            if self.finetuned_model_cfg.system_prompt is not None:
                # Prompted organism: the finetuned side sees the same rows
                # rendered with its system prompt. Rows are filtered jointly, so
                # the two lists are aligned and share position labels; only the
                # absolute positions differ.
                base_samples, ft_samples = load_and_tokenize_chat_dataset(
                    **loader_kwargs,
                    model_cfgs=[self.base_model_cfg, self.finetuned_model_cfg],
                )
            else:
                base_samples = ft_samples = load_and_tokenize_chat_dataset(**loader_kwargs)
            assert len(base_samples) == len(ft_samples)
            assert base_samples[0]["position_labels"] == ft_samples[0]["position_labels"]

            base_acts = extract_selected_positions_activations(
                model=self.base_model,
                samples=base_samples,
                layers=run_layers,
                batch_size=self.cfg.diffing.method.batch_size,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            self.clear_base_model()

            ft_acts = extract_selected_positions_activations(
                model=self.finetuned_model,
                samples=ft_samples,
                layers=run_layers,
                batch_size=self.cfg.diffing.method.batch_size,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            self.clear_finetuned_model()

            position_labels: List[int] = ft_samples[0]["position_labels"]
            num_positions = len(position_labels)
        else:
            first_n_tokens = load_and_tokenize_dataset(
                dataset_id,
                self.tokenizer,
                split=self.cfg.diffing.method.split,
                text_column=dataset_entry["text_column"],
                n=self.cfg.diffing.method.n,
                max_samples=self.cfg.diffing.method.max_samples,
                subset=dataset_entry.get("subset", None),
                streaming=dataset_entry.get("streaming", False),
                debug_print_samples=debug_print_samples,
                seed=seed,  # Note: shuffle not supported for streaming datasets
            )
            base_acts = extract_first_n_tokens_activations(
                self.base_model,
                first_n_tokens,
                run_layers,
                self.cfg.diffing.method.batch_size,
            )
            self.clear_base_model()

            ft_acts = extract_first_n_tokens_activations(
                self.finetuned_model,
                first_n_tokens,
                run_layers,
                self.cfg.diffing.method.batch_size,
            )
            self.clear_finetuned_model()

            position_labels = list(range(self.cfg.diffing.method.n))
            num_positions = len(position_labels)

        if norms_needed:
            self._compute_and_save_norms(
                dataset_id=dataset_id,
                run_layers=run_layers,
                base_acts=base_acts,
                ft_acts=ft_acts,
            )

        any_layer_for_meta = run_layers[0]
        num_sequences = ft_acts[any_layer_for_meta].shape[0]
        activation_dim = ft_acts[any_layer_for_meta].shape[2]

        for layer in list(layers_to_compute):
            diff = ft_acts[layer] - base_acts[layer]
            assert diff.shape[1] == num_positions and diff.shape[2] == activation_dim
            mean_diff = diff.mean(dim=0)
            base_mean = base_acts[layer].mean(dim=0)
            ft_mean = ft_acts[layer].mean(dim=0)
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            self._save_means_for_layer(
                out_dir,
                position_labels,
                mean_diff,
                base_mean,
                ft_mean,
                num_sequences,
                activation_dim,
            )

        return {
            "dataset_id": dataset_id,
            "run_layers": run_layers,
            "position_labels": position_labels,
            "aps_tasks_for_dataset": aps_tasks_for_dataset,
        }

    def analysis(self, ctx: Dict[str, Any]) -> None:
        """Run post-processing analysis on computed activation differences.

        Caches logit lens results and runs auto patch scope for each layer and position.

        Args:
            ctx: Context dict from compute_differences() containing:
                - "dataset_id": Dataset identifier
                - "run_layers": List of layer indices to process
                - "position_labels": List of position indices
                - "aps_tasks_for_dataset": Dict mapping layers to position sets for APS
        """
        dataset_id: str = str(ctx["dataset_id"]) if ("dataset_id" in ctx) else str(ctx)
        run_layers: List[int] = (
            list(ctx["run_layers"]) if ("run_layers" in ctx) else self.layers
        )
        position_labels: List[int] = (
            list(ctx["position_labels"])
            if ("position_labels" in ctx)
            else list(range(int(self.cfg.diffing.method.n)))
        )
        aps_tasks_for_dataset: Dict[int, set] = dict(
            ctx.get("aps_tasks_for_dataset", {})
        )
        if len(aps_tasks_for_dataset) == 0:
            run_layers, aps_tasks_for_dataset = self._get_run_layers_and_aps_tasks(
                dataset_id
            )

        # Cache logit lens (and, if configured, Jacobian lens) for each layer
        for layer in run_layers:
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            out_dir.mkdir(parents=True, exist_ok=True)
            self._cache_logit_lens_for_layer(out_dir, position_labels)
            self._cache_jacobian_lens_for_layer(out_dir, layer, position_labels)

        # Run auto Patchscope for each layer
        for layer in run_layers:
            out_dir = self.results_dir / f"layer_{layer}" / dataset_id.split("/")[-1]
            self._run_auto_patch_scope_for_layer(
                dataset_id, layer, out_dir, position_labels, aps_tasks_for_dataset
            )

    def visualize(self):
        visualize(self)

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available activation difference lens results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = defaultdict(dict)

        if not results_dir.exists():
            return results

        # Scan for activation difference lens results in the expected structure
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for organism_dir in model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                act_diff_lens_dir = organism_dir / "activation_difference_lens"

                # Check if there are any layer directories with results
                if act_diff_lens_dir.exists() and any(
                    layer_dir.is_dir()
                    and layer_dir.name.startswith("layer_")
                    and any(layer_dir.iterdir())
                    for layer_dir in act_diff_lens_dir.iterdir()
                ):
                    results[model_name][organism_name] = str(act_diff_lens_dir)

        return results

    def get_agent(self) -> BaseAgent:
        return ADLAgent(cfg=self.cfg)

    def get_baseline_agent(self) -> BaseAgent:
        return ADLBlackboxAgent(cfg=self.cfg)
