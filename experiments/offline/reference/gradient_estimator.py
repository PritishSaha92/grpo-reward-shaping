"""Estimate binary-vs-shaped local policy-gradient alignment for locked G7.

This is a final-checkpoint replay, not a reconstruction of historical training
updates. Under the locked PTA protocol it is a prospective reviewer-motivated
follow-up; the unlocked mode remains a post-hoc diagnostic. On a deterministic
sample of mixed MATH-500 n=8 groups it computes each rollout's completion-mean
score gradient with respect to the trainable LoRA parameters once, then
combines that same gradient with either the binary or deployed-shaped
normalized advantage. Shared KL and the head-learning auxiliary are
intentionally excluded so the comparison isolates the policy-gradient change
caused by reward shaping.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lavida.eval.provenance import (
    ordered_item_provenance,
    sha256_file,
    sha256_json,
    sha256_tree,
    verify_critical_source_closure,
)
from lavida.utils import configure_decoder_only_tokenizer
from scripts.analyze_inference_selection import _validate_frozen_plan


EPS = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument(
        "--base-model-revision",
        default=None,
        help="Pinned Hugging Face revision; required for release-grade provenance.",
    )
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--prompt-template", default="cot-4shot")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-groups", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--strict-provenance", action="store_true")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "protocols" / "pta2026_actionability_v1.yaml",
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=REPO_ROOT / "protocols" / "pta2026_actionability_v1.lock.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _recorded_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Replay provenance contains an empty artifact path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _locked_protocol(
    protocol_path: Path, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = _load_json(lock_path)
    if not isinstance(protocol, dict):
        raise TypeError("Protocol YAML must contain an object")
    if protocol.get("protocol_id") != "pta2026_actionability_v1":
        raise ValueError("Unexpected protocol_id")
    if lock.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Protocol lock ID mismatch")
    if lock.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Protocol YAML changed after locking")
    critical = lock.get("critical_file_sha256")
    if not isinstance(critical, dict):
        raise ValueError("Protocol lock lacks critical_file_sha256")
    for relative, expected_sha256 in critical.items():
        candidate = (REPO_ROOT / str(relative)).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"Locked critical path escapes repository: {relative}") from exc
        if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
            raise ValueError(f"Locked critical file is missing or changed: {relative}")
    verify_critical_source_closure(
        REPO_ROOT,
        lock.get("critical_source_roots", []),
        critical,
    )
    return protocol, lock


def _validate_derived_records(
    record_payload: dict[str, Any],
    result_payload: dict[str, Any],
    *,
    expected_results_sha256: str,
    num_groups: int,
    sample_seed: int,
    beta: float,
) -> list[dict[str, Any]]:
    """Validate and return derivation-ordered replay groups without resampling."""

    if record_payload.get("stage") != "locked_g7_policy_gradient_replay_record_derivation":
        raise ValueError("Records were not produced by the locked derivation stage")
    metadata = record_payload.get("metadata")
    groups = record_payload.get("groups")
    if not isinstance(metadata, dict) or not isinstance(groups, list):
        raise TypeError("Replay records require metadata and groups")
    if metadata.get("results_file_sha256") != expected_results_sha256:
        raise ValueError("Replay records do not match the supplied evaluation results")
    if int(metadata.get("num_groups", -1)) != num_groups or len(groups) != num_groups:
        raise ValueError("Replay record group count mismatch")
    if int(metadata.get("sample_seed", -1)) != sample_seed:
        raise ValueError("Replay record sample seed mismatch")
    if not math.isclose(float(metadata.get("beta")), beta, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Replay record beta mismatch")
    expected_ids = metadata.get("selected_group_ids")
    observed_ids = [str(group.get("group_id")) for group in groups]
    if expected_ids != observed_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("Replay record selected-group order/uniqueness mismatch")

    result_groups = result_payload.get("per_problem")
    if not isinstance(result_groups, list):
        raise TypeError("results JSON is missing per_problem")
    results_by_id = {str(group.get("unique_id")): group for group in result_groups}
    if len(results_by_id) != len(result_groups):
        raise ValueError("Evaluation results contain duplicate group IDs")
    for group in groups:
        group_id = str(group["group_id"])
        result = results_by_id.get(group_id)
        if result is None:
            raise ValueError(f"Replay group {group_id} is absent from results")
        rows = group.get("records")
        samples = result.get("samples")
        if not isinstance(rows, list) or not isinstance(samples, list):
            raise TypeError(f"Malformed replay/result group {group_id}")
        if len(rows) != len(samples) or len(rows) != 8:
            raise ValueError(f"Replay group {group_id} must contain exactly 8 rollouts")
        correctness = [bool(sample.get("correct")) for sample in samples]
        if not 0 < sum(correctness) < len(correctness):
            raise ValueError(f"Replay group {group_id} is not mixed")
        for sample_index, (row, sample) in enumerate(zip(rows, samples, strict=True)):
            if int(row.get("sample_idx", -1)) != sample_index:
                raise ValueError(f"Replay sample order mismatch for {group_id}")
            if not isinstance(sample.get("correct"), bool):
                raise TypeError(f"Result correctness is not Boolean for {group_id}")
            if row.get("correct") is not sample["correct"]:
                raise ValueError(f"Replay correctness changed for {group_id}")
            expected_coverage = not sample["correct"]
            if row.get("distance_defined") is not expected_coverage:
                raise ValueError(f"Replay distance coverage mismatch for {group_id}")
            if row.get("shaping_applied") is not expected_coverage:
                raise ValueError(f"Replay shaping gate mismatch for {group_id}")
            distance = float(row.get("deployed_distance"))
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(f"Invalid deployed distance for {group_id}")
            if sample["correct"] and distance != 0.0:
                raise ValueError(f"Correct rollout has nonzero deployed distance: {group_id}")
            token_ids = sample.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"Saved completion token IDs are required for {group_id}")
        prompt_ids = result.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"Saved prompt token IDs are required for {group_id}")
    return groups


def validate_replay_inputs(
    *,
    records_path: Path,
    results_path: Path,
    checkpoint_path: Path,
    protocol_path: Path,
    protocol_lock_path: Path,
    num_groups: int,
    sample_seed: int,
    beta: float,
    base_model: str,
    base_model_revision: str | None,
    tokenizer_id: str | None,
    tokenizer_revision: str | None,
    dataset_revision: str | None,
    prompt_template: str,
    max_model_len: int,
    strict_provenance: bool,
) -> dict[str, Any]:
    """CPU preflight for all replay artifacts and immutable pins."""

    records = _load_json(records_path)
    results = _load_json(results_path)
    groups = _validate_derived_records(
        records,
        results,
        expected_results_sha256=sha256_file(results_path),
        num_groups=num_groups,
        sample_seed=sample_seed,
        beta=beta,
    )
    protocol = lock = None
    if strict_provenance:
        protocol, lock = _locked_protocol(protocol_path, protocol_lock_path)
        replay = protocol["coverage_advantage_novelty"]["policy_gradient_replay"]
        decode = protocol["pinned_redecode"]
        pins = protocol["pins"]
        resolved = {
            "base_model": pins["base_model"]["id"],
            "base_model_revision": pins["base_model"]["revision"],
            "tokenizer_id": pins["tokenizer"]["id"],
            "tokenizer_revision": pins["tokenizer"]["revision"],
            "dataset_revision": pins["math500"]["revision"],
        }
        supplied = {
            "base_model": base_model,
            "base_model_revision": base_model_revision or resolved["base_model_revision"],
            "tokenizer_id": tokenizer_id or resolved["tokenizer_id"],
            "tokenizer_revision": tokenizer_revision or resolved["tokenizer_revision"],
            "dataset_revision": dataset_revision or resolved["dataset_revision"],
        }
        if supplied != resolved:
            raise ValueError(f"Replay model/tokenizer/dataset pins mismatch: {supplied!r}")
        if int(replay["groups_per_decoder_seed"]) != num_groups:
            raise ValueError("num_groups differs from locked replay")
        if int(replay["group_sample_seed"]) != sample_seed:
            raise ValueError("sample_seed differs from locked replay")
        locked_beta = float(
            protocol["training_arms"]["g7"]["reward_shaping"]["beta"]
        )
        if not math.isclose(beta, locked_beta, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("beta differs from locked G7 reward shaping")
        if prompt_template != decode["prompt_template"]:
            raise ValueError("prompt template differs from locked replay")
        if max_model_len != int(decode["max_model_len"]):
            raise ValueError("max_model_len differs from locked replay")
        if replay["generator_and_policy"] != "historical_G7_checkpoint":
            raise ValueError("Locked replay does not specify historical G7")
        expected_replay_semantics = {
            "eligible_groups": "mixed_outcome_only",
            "rollout_score": "completion_mean_log_probability",
            "parameters": "trainable_LoRA_only",
            "compare": "binary_normalized_advantage_vs_deployed_shaped_normalized_advantage",
            "exclude_from_both_directions": ["KL_term", "learned_head_auxiliary"],
        }
        observed_replay_semantics = {
            key: replay.get(key) for key in expected_replay_semantics
        }
        if observed_replay_semantics != expected_replay_semantics:
            raise ValueError("Locked replay semantics differ from this estimator")
        metadata = records["metadata"]
        plan_path = _recorded_path(metadata.get("plan"))
        matrices_path = _recorded_path(metadata.get("matrices"))
        if metadata.get("plan_file_sha256") != sha256_file(plan_path):
            raise ValueError("Frozen selector-plan artifact changed after derivation")
        if metadata.get("matrices_file_sha256") != sha256_file(matrices_path):
            raise ValueError("Geometry artifact changed after derivation")
        frozen_plan, _ = _validate_frozen_plan(
            _load_json(plan_path),
            matrices_path=matrices_path,
            results_path=results_path,
            strict_provenance=True,
        )
        if sha256_json(frozen_plan) != metadata.get("frozen_plan_sha256"):
            raise ValueError("Replay records contain a different frozen plan")
        comparisons = {
            "strict_records": (metadata.get("strict_provenance"), True),
            "protocol": (metadata.get("protocol_file_sha256"), sha256_file(protocol_path)),
            "protocol_lock": (
                metadata.get("protocol_lock_file_sha256"),
                sha256_file(protocol_lock_path),
            ),
            "critical_manifest": (
                metadata.get("critical_manifest_sha256"),
                sha256_json(lock["critical_file_sha256"]),
            ),
            "generator_arm": (str(metadata.get("generator_arm", "")).upper(), "G7"),
            "decoder_seed_scheduled": (
                int(metadata.get("decoder_seed")) in {int(x) for x in replay["decoder_seeds"]},
                True,
            ),
            "base_model": (metadata.get("base_model"), resolved["base_model"]),
            "base_model_revision": (
                metadata.get("base_model_revision"),
                resolved["base_model_revision"],
            ),
            "tokenizer_id": (metadata.get("tokenizer_id"), resolved["tokenizer_id"]),
            "tokenizer_revision": (
                metadata.get("tokenizer_revision"),
                resolved["tokenizer_revision"],
            ),
            "dataset_revision": (
                metadata.get("dataset_revision"),
                resolved["dataset_revision"],
            ),
            "dataset_fingerprint": (
                metadata.get("dataset_fingerprint"),
                pins["math500"]["fingerprint"],
            ),
            "ordered_ids": (
                metadata.get("ordered_example_ids_sha256"),
                pins["math500"]["ordered_example_ids_sha256"],
            ),
            "checkpoint": (
                sha256_tree(checkpoint_path)["sha256"],
                pins["artifacts"]["historical_checkpoints"]["G7"]["tree_sha256"],
            ),
            "record_generator_checkpoint": (
                metadata.get("generator_checkpoint_sha256"),
                pins["artifacts"]["historical_checkpoints"]["G7"]["tree_sha256"],
            ),
            "record_scorer_checkpoint": (
                metadata.get("scorer_checkpoint_sha256"),
                pins["artifacts"]["historical_checkpoints"]["G7"]["tree_sha256"],
            ),
        }
        mismatches = [
            f"{name}: observed={observed!r}, expected={expected!r}"
            for name, (observed, expected) in comparisons.items()
            if observed != expected
        ]
        if mismatches:
            raise ValueError("Strict replay provenance mismatch: " + "; ".join(mismatches))
    else:
        resolved = {
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "tokenizer_id": tokenizer_id or base_model,
            "tokenizer_revision": tokenizer_revision or base_model_revision,
            "dataset_revision": dataset_revision,
        }
    return {
        "record_payload": records,
        "result_payload": results,
        "groups": groups,
        "protocol": protocol,
        "lock": lock,
        "resolved": resolved,
        "checkpoint_inventory": sha256_tree(checkpoint_path),
    }


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std(ddof=0) + EPS)


def _prepare_sequence(
    prompt_ids: list[int], completion_ids: list[int], max_model_len: int
) -> tuple[list[int], int]:
    if len(completion_ids) >= max_model_len:
        raise ValueError(f"Completion length {len(completion_ids)} exceeds replay limit")
    keep_prompt = max_model_len - len(completion_ids)
    dropped = max(0, len(prompt_ids) - keep_prompt)
    return prompt_ids[-keep_prompt:] + completion_ids, dropped


def _cosine_and_ratio(
    binary_gradient: dict[str, torch.Tensor],
    shaped_gradient: dict[str, torch.Tensor],
) -> dict[str, float]:
    dot = torch.zeros((), device="cuda", dtype=torch.float64)
    binary_sq = torch.zeros_like(dot)
    shaped_sq = torch.zeros_like(dot)
    for name in binary_gradient:
        binary = binary_gradient[name].double()
        shaped = shaped_gradient[name].double()
        dot += torch.sum(binary * shaped)
        binary_sq += torch.sum(binary * binary)
        shaped_sq += torch.sum(shaped * shaped)
    dot_value = float(dot.item())
    binary_norm = math.sqrt(float(binary_sq.item()))
    shaped_norm = math.sqrt(float(shaped_sq.item()))
    denominator = binary_norm * shaped_norm
    return {
        "cosine": dot_value / denominator if denominator else math.nan,
        "norm_ratio_shaped_to_binary": (
            shaped_norm / binary_norm if binary_norm else math.nan
        ),
        "binary_norm": binary_norm,
        "shaped_norm": shaped_norm,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {args.output}")
    if args.num_groups <= 0:
        raise ValueError("num_groups must be positive")

    started = time.time()
    preflight = validate_replay_inputs(
        records_path=args.records,
        results_path=args.results,
        checkpoint_path=args.checkpoint,
        protocol_path=args.protocol,
        protocol_lock_path=args.protocol_lock,
        num_groups=args.num_groups,
        sample_seed=args.sample_seed,
        beta=args.beta,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        dataset_revision=args.dataset_revision,
        prompt_template=args.prompt_template,
        max_model_len=args.max_model_len,
        strict_provenance=args.strict_provenance,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lavida.eval.evaluate_math import load_benchmark

    resolved = preflight["resolved"]
    result_payload = preflight["result_payload"]
    result_groups_by_id = {
        str(group["unique_id"]): group for group in result_payload["per_problem"]
    }
    replay_groups = preflight["groups"]
    selected = [result_groups_by_id[str(group["group_id"])] for group in replay_groups]

    # Load the immutable dataset revision even though replay uses the exact
    # prompt token IDs saved by the pinned decode.  This independently checks
    # the dataset fingerprint/order pin rather than silently trusting a cache.
    benchmark, benchmark_metadata = load_benchmark(
        "math500", revision=resolved["dataset_revision"], return_metadata=True
    )
    dataset_order = ordered_item_provenance(benchmark)
    if args.strict_provenance:
        pins = preflight["protocol"]["pins"]["math500"]
        if benchmark_metadata.get("fingerprint") != pins["fingerprint"]:
            raise ValueError("Loaded MATH-500 fingerprint differs from the lock")
        if dataset_order["ordered_item_ids_sha256"] != pins["ordered_example_ids_sha256"]:
            raise ValueError("Loaded MATH-500 order differs from the lock")

    _validated_tokenizer = configure_decoder_only_tokenizer(
        AutoTokenizer.from_pretrained(
            resolved["tokenizer_id"],
            revision=resolved["tokenizer_revision"],
            trust_remote_code=True,
        )
    )
    prompt_ids_by_group = [
        [int(token) for token in group["prompt_token_ids"]] for group in selected
    ]

    base = AutoModelForCausalLM.from_pretrained(
        resolved["base_model"],
        revision=resolved["base_model_revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(args.checkpoint), is_trainable=True)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")
    binary_gradient = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
        for name, parameter in trainable.items()
    }
    shaped_gradient = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
        for name, parameter in trainable.items()
    }

    prefix_targets = sorted({min(8, args.num_groups), min(16, args.num_groups), args.num_groups})
    prefix_metrics: dict[str, dict[str, float]] = {}
    completion_token_counts: list[int] = []
    truncation_count = 0
    max_dropped = 0
    advantage_binary_all: list[float] = []
    advantage_shaped_all: list[float] = []

    for group_number, (group, replay_group, prompt_ids) in enumerate(
        zip(selected, replay_groups, prompt_ids_by_group, strict=True), start=1
    ):
        group_id = str(group["unique_id"])
        if str(replay_group["group_id"]) != group_id:
            raise ValueError("Replay/result group order changed after preflight")
        geometry = replay_group["records"]
        if len(geometry) != len(group["samples"]):
            raise ValueError(f"Geometry/result row mismatch for {group_id}")
        rewards = np.asarray(
            [float(sample["correct"]) for sample in group["samples"]],
            dtype=np.float64,
        )
        distances = np.asarray(
            [float(row["deployed_distance"]) for row in geometry],
            dtype=np.float64,
        )
        binary_advantage = _normalise(rewards)
        shaped_advantage = _normalise(rewards - args.beta * distances)
        advantage_binary_all.extend(binary_advantage.tolist())
        advantage_shaped_all.extend(shaped_advantage.tolist())

        for sample_index, sample in enumerate(group["samples"]):
            completion_ids = [int(token) for token in sample["token_ids"]]
            full_ids, dropped = _prepare_sequence(
                prompt_ids, completion_ids, args.max_model_len
            )
            truncation_count += int(dropped > 0)
            max_dropped = max(max_dropped, dropped)
            completion_token_counts.append(len(completion_ids))

            ids = torch.as_tensor(full_ids, dtype=torch.long, device="cuda").unsqueeze(0)
            attention_mask = torch.ones_like(ids)
            outputs = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            completion_len = len(completion_ids)
            logits = outputs.logits[:, :-1, :][:, -completion_len:, :]
            targets = torch.as_tensor(
                completion_ids, dtype=torch.long, device=logits.device
            ).unsqueeze(0)
            mean_log_probability = -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            )
            model.zero_grad(set_to_none=True)
            mean_log_probability.backward()

            binary_weight = float(binary_advantage[sample_index])
            shaped_weight = float(shaped_advantage[sample_index])
            with torch.no_grad():
                for name, parameter in trainable.items():
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad.detach().float()
                    binary_gradient[name].add_(gradient, alpha=binary_weight)
                    shaped_gradient[name].add_(gradient, alpha=shaped_weight)
            del outputs, logits, mean_log_probability

        if group_number in prefix_targets:
            prefix_metrics[str(group_number)] = _cosine_and_ratio(
                binary_gradient, shaped_gradient
            )
        print(
            f"policy-gradient replay {group_number}/{args.num_groups} groups",
            flush=True,
        )

    binary_advantage_array = np.asarray(advantage_binary_all)
    shaped_advantage_array = np.asarray(advantage_shaped_all)
    advantage_cosine = float(
        np.dot(binary_advantage_array, shaped_advantage_array)
        / (
            np.linalg.norm(binary_advantage_array)
            * np.linalg.norm(shaped_advantage_array)
        )
    )
    output = {
        "schema_version": 1,
        "stage": "final_checkpoint_policy_gradient_replay",
        "protocol_id": "pta2026_actionability_v1",
        "metadata": {
            "records": str(args.records),
            "records_sha256": sha256_file(args.records),
            "results": str(args.results),
            "results_sha256": sha256_file(args.results),
            "checkpoint": str(args.checkpoint),
            "checkpoint_tree_sha256": preflight["checkpoint_inventory"]["sha256"],
            "checkpoint_tree_hash_schema": "lavida.eval.provenance.sha256_tree.v1",
            "protocol_file_sha256": (
                sha256_file(args.protocol) if args.strict_provenance else None
            ),
            "protocol_lock_file_sha256": (
                sha256_file(args.protocol_lock) if args.strict_provenance else None
            ),
            "critical_manifest_sha256": (
                sha256_json(preflight["lock"]["critical_file_sha256"])
                if args.strict_provenance
                else None
            ),
            "base_model": resolved["base_model"],
            "base_model_revision": resolved["base_model_revision"],
            "tokenizer_id": resolved["tokenizer_id"],
            "tokenizer_revision": resolved["tokenizer_revision"],
            "dataset_revision": resolved["dataset_revision"],
            "dataset_fingerprint": benchmark_metadata.get("fingerprint"),
            "ordered_example_ids_sha256": dataset_order[
                "ordered_item_ids_sha256"
            ],
            "beta": args.beta,
            "sample_seed": args.sample_seed,
            "n_groups": args.num_groups,
            "n_rollouts": args.num_groups * 8,
            "decoder_seed": preflight["record_payload"]["metadata"].get(
                "decoder_seed"
            ),
            "selected_group_ids": [str(group["unique_id"]) for group in selected],
            "selected_n_correct_distribution": dict(
                Counter(str(int(group["n_correct"])) for group in selected)
            ),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in trainable.values())
            ),
            "completion_tokens_mean": float(np.mean(completion_token_counts)),
            "completion_tokens_total": int(sum(completion_token_counts)),
            "teacher_forcing_context_truncation": {
                "n_rollouts": truncation_count,
                "max_dropped_prompt_tokens": max_dropped,
            },
            "hardware": torch.cuda.get_device_name(0),
            "wall_clock_seconds": time.time() - started,
            "strict_provenance": args.strict_provenance,
            "post_hoc": not args.strict_provenance,
            "analysis_timing": (
                "prospective_reviewer_motivated_followup_under_locked_protocol"
                if args.strict_provenance
                else "post_hoc_diagnostic"
            ),
        },
        "policy_gradient": {
            "estimator": (
                "on-policy unclipped completion-mean score-function gradient on LoRA "
                "parameters at the final checkpoint; shared KL and head auxiliary excluded"
            ),
            "prefix_group_estimates": prefix_metrics,
            "final": prefix_metrics[str(args.num_groups)],
            "selected_subset_advantage_cosine": advantage_cosine,
            "scope_warning": (
                "Final-checkpoint post-hoc replay on a fixed sampled subset, not a "
                "reconstruction of the 2,000 historical training updates."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
