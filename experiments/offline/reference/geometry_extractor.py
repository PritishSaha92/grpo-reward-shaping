"""Extract fixed-G7-scorer audit records and label-free pairwise geometry.

This is a forward-only GPU diagnostic for a saved MATH-500 n=8 re-decode from
any scheduled A/D/G/G7 generator.  Every candidate is teacher-forced through
one restored historical G7 policy/checkpoint and its frozen VAE/projection/MRN
heads so geometry is measured in one common scorer coordinate system.
It reconstructs the training feature path (last-four-layer average -> four
trajectory statistics -> frozen VAE -> learned projection/MRN heads) and saves
both:

* ``diagnostic_distance``: distance to the nearest *other* correct rollout,
  when that quantity is defined; and
* ``deployed_distance``: the quantity the historical G7 trainer subtracted
  (zero for correct rollouts and degenerate groups); for A/D/G candidate sets
  this retained field is explicitly counterfactual, not a generator reward.

It also saves, for every prompt (including all-wrong and all-correct groups),
the complete candidate-by-candidate matrices for learned MRN distance,
projected-space L2, and raw frozen-latent L2.  All three metric matrices are
symmetric.  This is distinct from the deployed reward's outcome-gated
asymmetry, which penalises only incorrect queries.  Those matrices are constructed
only from candidate representations: correctness, the gold answer, and an
oracle-selected reference set are not inputs to the matrix helper.  The
separate correctness-conditioned records are retained for backward
compatibility with the coverage--advantage audit.

Keeping these fields separate prevents the oracle-conditioned outcome-AUROC
diagnostic from being mistaken for the asymmetric reward GRPO consumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lavida.models.ae import FrozenAE  # noqa: E402
from lavida.eval.provenance import sha256_file, sha256_json, sha256_tree  # noqa: E402
from lavida.models.mrn_head import MRNDistance  # noqa: E402
from lavida.models.projection_head import ProjectionHead  # noqa: E402
from lavida.models.utils import average_last_layers, build_psi_from_hbar  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-ae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument(
        "--base-model-revision",
        default=None,
        help="Pinned Hugging Face revision; required for release-grade provenance.",
    )
    parser.add_argument("--prompt-template", default="cot-4shot")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-layers-averaged", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--expected-group-size",
        type=int,
        default=8,
        help="Require exactly this many candidates per prompt (default: 8).",
    )
    parser.add_argument(
        "--expected-groups",
        type=int,
        default=500,
        help="Require this many prompt groups (default: all 500 MATH-500 prompts).",
    )
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help=(
            "Require locked-protocol run metadata and saved prompt token IDs, "
            "and verify model/checkpoint/VAE pins before extraction."
        ),
    )
    parser.add_argument(
        "--pairwise-only",
        action="store_true",
        help=(
            "Write label-free matrices without constructing the legacy "
            "correctness-conditioned rollout records."
        ),
    )
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


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch<2.0 compatibility for local inspection.
        return torch.load(path, map_location="cpu")


def _load_result_groups(
    path: Path, *, expected_group_size: int, expected_groups: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("per_problem")
    if not isinstance(groups, list):
        raise TypeError(f"{path} does not contain a per_problem list")
    if len(groups) != expected_groups:
        raise ValueError(
            f"{path} has {len(groups)} prompt groups; expected {expected_groups}"
        )
    for group in groups:
        samples = group.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Every per_problem row must contain non-empty samples")
        if len(samples) != expected_group_size:
            raise ValueError(
                f"Group {group.get('unique_id')!r} has {len(samples)} samples; "
                f"expected {expected_group_size}"
            )
        for sample in samples:
            token_ids = sample.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(
                    "Saved token_ids are required; re-run evaluation with save_token_ids"
                )
    return payload, groups


def _load_locked_protocol(
    protocol_path: Path, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(lock, dict):
        raise TypeError("Protocol and lock files must contain objects")
    if protocol.get("protocol_id") != "pta2026_actionability_v1":
        raise ValueError("Unexpected actionability protocol_id")
    if lock.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Protocol lock ID does not match the protocol")
    actual_protocol_sha = sha256_file(protocol_path)
    if lock.get("protocol_sha256") != actual_protocol_sha:
        raise ValueError("Protocol YAML changed after its lock was written")
    critical = lock.get("critical_file_sha256")
    expected_script_sha = (
        critical.get("scripts/extract_g7_coverage_advantage_records.py")
        if isinstance(critical, dict)
        else None
    )
    actual_script_sha = sha256_file(Path(__file__).resolve())
    if expected_script_sha != actual_script_sha:
        raise ValueError("Extractor code does not match the locked critical-file hash")
    return protocol, lock


def _validate_strict_source_provenance(
    payload: dict[str, Any],
    *,
    scorer_checkpoint_sha256: str,
    frozen_ae_sha256: str,
    base_model: str,
    base_model_revision: str | None,
    protocol: dict[str, Any],
    protocol_lock: dict[str, Any] | None = None,
    protocol_lock_sha256: str | None = None,
) -> dict[str, Any]:
    run_metadata = payload.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError("Strict extraction requires a run_metadata object")
    required = (
        "protocol_id",
        "arm_label",
        "checkpoint_tree_sha256",
        "frozen_vae_sha256",
        "code_commit",
        "code_dirty",
        "container_digest",
        "modal_image_id",
        "qwen_eval_tree_sha256",
        "protocol_lock_sha256",
        "critical_manifest_sha256",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dataset_id",
        "dataset_revision",
        "dataset_fingerprint",
        "ordered_example_ids_sha256",
        "prompt_template_sha256",
        "engine_seed",
        "decoder_seed",
        "exact_sampling_config",
    )
    missing = [key for key in required if run_metadata.get(key) in (None, "")]
    if missing:
        raise ValueError(
            "Strict extraction source is missing run metadata: " + ", ".join(missing)
        )
    if run_metadata["protocol_id"] != protocol["protocol_id"]:
        raise ValueError(
            f"Unexpected protocol_id {run_metadata['protocol_id']!r}; "
            "expected 'pta2026_actionability_v1'"
        )
    arm = str(run_metadata["arm_label"]).upper()
    scheduled_arms = [
        str(value).upper() for value in protocol["pinned_redecode"]["arms"]
    ]
    if arm not in scheduled_arms:
        raise ValueError(
            f"Geometry source arm {arm!r} is not scheduled by the protocol"
        )
    if run_metadata["code_dirty"] is not False:
        raise ValueError("Strict extraction requires a clean source code snapshot")
    pins = protocol["pins"]
    expected_generator_sha = pins["artifacts"]["historical_checkpoints"][arm][
        "tree_sha256"
    ]
    expected_scorer_sha = pins["artifacts"]["historical_checkpoints"]["G7"][
        "tree_sha256"
    ]
    comparisons = {
        "generator_checkpoint_sha256": (
            str(run_metadata["checkpoint_tree_sha256"]),
            str(expected_generator_sha),
        ),
        "scorer_checkpoint_sha256": (
            scorer_checkpoint_sha256,
            str(expected_scorer_sha),
        ),
        "frozen_vae_sha256": (
            str(run_metadata["frozen_vae_sha256"]),
            frozen_ae_sha256,
        ),
        "frozen_vae_protocol": (
            frozen_ae_sha256,
            str(pins["artifacts"]["frozen_vae"]["sha256"]),
        ),
        "model_id": (str(run_metadata["model_id"]), base_model),
        "model_id_protocol": (base_model, str(pins["base_model"]["id"])),
        "model_revision": (
            str(run_metadata["model_revision"]),
            str(base_model_revision),
        ),
        "container_digest": (
            str(run_metadata["container_digest"]),
            str(pins["container"]["digest"]),
        ),
        "modal_image_id": (
            str(run_metadata["modal_image_id"]),
            str(pins["container"]["modal_image_id"]),
        ),
        "model_revision_protocol": (
            str(run_metadata["model_revision"]),
            str(pins["base_model"]["revision"]),
        ),
        "tokenizer_revision": (
            str(run_metadata["tokenizer_revision"]),
            str(pins["tokenizer"]["revision"]),
        ),
        "dataset_revision": (
            str(run_metadata["dataset_revision"]),
            str(pins["math500"]["revision"]),
        ),
        "dataset_fingerprint": (
            str(run_metadata["dataset_fingerprint"]),
            str(pins["math500"]["fingerprint"]),
        ),
        "ordered_example_ids_sha256": (
            str(run_metadata["ordered_example_ids_sha256"]),
            str(pins["math500"]["ordered_example_ids_sha256"]),
        ),
        "prompt_template_sha256": (
            str(run_metadata["prompt_template_sha256"]),
            str(pins["prompt_template"]["implementation_sha256"]),
        ),
        "qwen_eval_tree_sha256": (
            str(run_metadata["qwen_eval_tree_sha256"]),
            str(pins["qwen_eval"]["tree_sha256"]),
        ),
    }
    if protocol_lock is not None:
        comparisons["code_commit"] = (
            str(run_metadata["code_commit"]),
            str(protocol_lock["git_commit"]),
        )
        comparisons["critical_manifest_sha256"] = (
            str(run_metadata["critical_manifest_sha256"]),
            sha256_json(protocol_lock["critical_file_sha256"]),
        )
    if protocol_lock_sha256 is not None:
        comparisons["protocol_lock_sha256"] = (
            str(run_metadata["protocol_lock_sha256"]),
            protocol_lock_sha256,
        )
    mismatches = [
        f"{name}: source={source!r}, local={local!r}"
        for name, (source, local) in comparisons.items()
        if source != local
    ]
    if mismatches:
        raise ValueError("Strict provenance mismatch: " + "; ".join(mismatches))
    if (
        arm == "G7"
        and str(run_metadata["checkpoint_tree_sha256"]) != scorer_checkpoint_sha256
    ):
        raise ValueError(
            "G7 generator and fixed G7 scorer checkpoint hashes must match"
        )
    sampling = run_metadata["exact_sampling_config"]
    if not isinstance(sampling, dict):
        raise TypeError("exact_sampling_config must be an object")
    decode = protocol["pinned_redecode"]
    if int(run_metadata["engine_seed"]) != int(decode["engine_seed"]):
        raise ValueError("Source engine_seed does not match the locked protocol")
    if int(run_metadata["decoder_seed"]) not in {
        int(seed) for seed in decode["decoder_seeds"]
    }:
        raise ValueError("Source decoder_seed is not scheduled by the locked protocol")
    expected_sampling = {
        "n": decode["num_samples"],
        "temperature": decode["temperature"],
        "max_tokens": decode["max_tokens"],
        "top_p": decode["top_p"],
        "top_k": decode["top_k"],
        "repetition_penalty": decode["repetition_penalty"],
        "logprobs": 0,
        "generation_details_saved": True,
    }
    sampling_mismatches = [
        f"{key}: source={sampling.get(key)!r}, expected={expected!r}"
        for key, expected in expected_sampling.items()
        if sampling.get(key) != expected
    ]
    if sampling_mismatches:
        raise ValueError("Strict sampling mismatch: " + "; ".join(sampling_mismatches))
    return run_metadata


def _validate_strict_completion_fields(
    groups: list[dict[str, Any]], *, validate_correctness: bool
) -> None:
    required = [
        "sample_index",
        "text",
        "token_ids",
        "cumulative_logprob",
        "mean_logprob",
        "parsed_answer",
    ]
    if validate_correctness:
        required.append("correct")
    for group in groups:
        group_id = str(group.get("unique_id"))
        prompt_ids = group.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"{group_id} lacks saved prompt_token_ids")
        for sample_index, sample in enumerate(group["samples"]):
            missing = [key for key in required if key not in sample]
            if missing:
                raise ValueError(
                    f"{group_id} sample {sample_index} lacks: {', '.join(missing)}"
                )
            if int(sample["sample_index"]) != sample_index:
                raise ValueError(
                    f"{group_id} sample_index does not match its list position"
                )
            if not isinstance(sample["text"], str):
                raise TypeError(
                    f"{group_id} sample {sample_index} text must be a string"
                )
            if validate_correctness and not isinstance(sample["correct"], bool):
                raise TypeError(
                    f"{group_id} sample {sample_index} correct must be Boolean"
                )
            for field in ("cumulative_logprob", "mean_logprob"):
                try:
                    value = float(sample[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{group_id} sample {sample_index} {field} must be numeric"
                    ) from exc
                if not torch.isfinite(torch.tensor(value)):
                    raise ValueError(
                        f"{group_id} sample {sample_index} {field} must be finite"
                    )


def _group_kind(samples: list[dict[str, Any]]) -> str:
    n_correct = sum(bool(sample["correct"]) for sample in samples)
    if n_correct == 0:
        return "zero_correct"
    if n_correct == len(samples):
        return "all_correct"
    return "mixed"


def _prepare_sequence(
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    max_model_len: int,
) -> tuple[list[int], int]:
    if len(completion_ids) >= max_model_len:
        raise ValueError(
            f"Completion length {len(completion_ids)} leaves no prompt context at "
            f"max_model_len={max_model_len}"
        )
    keep_prompt = max_model_len - len(completion_ids)
    dropped = max(0, len(prompt_ids) - keep_prompt)
    return prompt_ids[-keep_prompt:] + completion_ids, dropped


@torch.inference_mode()
def _project_completion(
    *,
    model,
    frozen_ae: FrozenAE,
    projection_head: ProjectionHead,
    input_ids: list[int],
    completion_len: int,
    num_layers_averaged: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(ids)
    outputs = model(
        input_ids=ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    h_bar = average_last_layers(
        outputs.hidden_states, num_layers_averaged=num_layers_averaged
    )
    # This mirrors LaViDATrainer._get_per_token_logps_and_entropies: hidden
    # states are shifted to the token-prediction surface, then the completion
    # span is selected.
    h_bar = h_bar[:, :-1, :]
    h_bar = h_bar[:, -completion_len:, :]
    completion_mask = torch.ones((1, completion_len), dtype=torch.long, device=device)
    psi = build_psi_from_hbar(h_bar, completion_mask)
    latent = frozen_ae.encode(psi)
    projected = projection_head(latent.float())
    return (
        latent.squeeze(0).detach().float().cpu(),
        projected.squeeze(0).detach().float().cpu(),
    )


@torch.inference_mode()
def _pairwise_distance_matrices(
    *,
    raw_latents: torch.Tensor,
    projected_latents: torch.Tensor,
    mrn: MRNDistance,
) -> dict[str, torch.Tensor]:
    """Construct three label-free candidate-distance matrices.

    This helper deliberately has no correctness or reference-set argument.
    The learned head may have been trained with outcome supervision, but the
    current evaluation labels do not participate in matrix construction.
    Diagonals are set to exact zero by convention; ``MRNDistance`` otherwise
    has a small positive self-distance because of its numerical epsilon.
    """

    if raw_latents.ndim != 2 or projected_latents.ndim != 2:
        raise ValueError("raw_latents and projected_latents must be rank-2")
    if raw_latents.shape[0] != projected_latents.shape[0]:
        raise ValueError("raw/projected candidate counts must match")
    if projected_latents.shape[1] != mrn.dim:
        raise ValueError(
            f"projected dimension {projected_latents.shape[1]} != MRN dim {mrn.dim}"
        )
    if raw_latents.shape[0] == 0:
        raise ValueError("at least one candidate is required")

    raw_latents = raw_latents.float()
    projected_latents = projected_latents.float()
    n_candidates = int(projected_latents.shape[0])
    left = projected_latents[:, None, :].expand(
        n_candidates, n_candidates, projected_latents.shape[1]
    )
    right = projected_latents[None, :, :].expand_as(left)
    learned_mrn = mrn(left, right)
    projected_l2 = torch.cdist(projected_latents, projected_latents, p=2)
    raw_l2 = torch.cdist(raw_latents, raw_latents, p=2)

    matrices = {
        "mrn": learned_mrn,
        "projected_l2": projected_l2,
        "raw_l2": raw_l2,
    }
    for name, matrix in matrices.items():
        if not torch.isfinite(matrix).all():
            raise FloatingPointError(f"{name} contains a non-finite value")
        matrix.fill_diagonal_(0.0)
        if not torch.allclose(matrix, matrix.T, atol=1e-5, rtol=1e-6):
            raise ValueError(
                f"{name} is not symmetric; MRN metric symmetry and L2 symmetry "
                "are required by the locked protocol"
            )
    return matrices


def _strict_load_heads(
    checkpoint: Path, device: torch.device
) -> tuple[ProjectionHead, MRNDistance, dict[str, Any]]:
    state_path = checkpoint / "lavida_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = _torch_load(state_path)
    projection_dim = int(state.get("cr_projection_dim", 256))
    projection_hidden = int(state.get("cr_projection_hidden", 512))
    projection = ProjectionHead(
        input_dim=256,
        hidden_dim=projection_hidden,
        output_dim=projection_dim,
    )
    mrn = MRNDistance(dim=projection_dim)
    projection.load_state_dict(state["cr_projection_head"], strict=True)
    mrn.load_state_dict(state["cr_mrn_head"], strict=True)
    projection.to(device=device, dtype=torch.float32).eval()
    mrn.to(device=device, dtype=torch.float32).eval()
    return projection, mrn, state


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU")
    if args.beta < 0.0:
        raise ValueError("beta must be non-negative")
    if args.expected_group_size <= 0:
        raise ValueError("expected-group-size must be positive")
    if args.expected_groups <= 0:
        raise ValueError("expected-groups must be positive")
    if args.strict_provenance and args.output.exists():
        raise FileExistsError(
            f"Strict extraction refuses to overwrite existing output: {args.output}"
        )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    device = torch.device("cuda")
    payload, groups = _load_result_groups(
        args.results,
        expected_group_size=args.expected_group_size,
        expected_groups=args.expected_groups,
    )
    scorer_checkpoint_inventory = sha256_tree(args.checkpoint)
    scorer_checkpoint_sha256 = str(scorer_checkpoint_inventory["sha256"])
    frozen_ae_sha256 = sha256_file(args.frozen_ae)
    source_run_metadata = None
    protocol = None
    if args.strict_provenance:
        protocol, protocol_lock = _load_locked_protocol(
            args.protocol, args.protocol_lock
        )
        decode = protocol["pinned_redecode"]
        strict_argument_mismatches = []
        for name, actual, expected in (
            ("expected_group_size", args.expected_group_size, decode["num_samples"]),
            ("max_model_len", args.max_model_len, decode["max_model_len"]),
            ("prompt_template", args.prompt_template, decode["prompt_template"]),
        ):
            if actual != expected:
                strict_argument_mismatches.append(
                    f"{name}: argument={actual!r}, protocol={expected!r}"
                )
        if strict_argument_mismatches:
            raise ValueError(
                "Strict extractor argument mismatch: "
                + "; ".join(strict_argument_mismatches)
            )
        source_run_metadata = _validate_strict_source_provenance(
            payload,
            scorer_checkpoint_sha256=scorer_checkpoint_sha256,
            frozen_ae_sha256=frozen_ae_sha256,
            base_model=args.base_model,
            base_model_revision=args.base_model_revision,
            protocol=protocol,
            protocol_lock=protocol_lock,
            protocol_lock_sha256=sha256_file(args.protocol_lock),
        )
        _validate_strict_completion_fields(
            groups, validate_correctness=not args.pairwise_only
        )

    if all(
        isinstance(group.get("prompt_token_ids"), list)
        and bool(group["prompt_token_ids"])
        for group in groups
    ):
        prompt_token_ids = [
            [int(token) for token in group["prompt_token_ids"]] for group in groups
        ]
        prompt_source = "saved_result_prompt_token_ids"
    else:
        if args.strict_provenance:
            raise ValueError(
                "Strict extraction requires prompt_token_ids on every per_problem row"
            )
        from lavida.eval.evaluate_math import build_prompts, load_benchmark
        from lavida.utils import configure_decoder_only_tokenizer

        benchmark = load_benchmark("math500")
        benchmark_by_id = {str(item["unique_id"]): item for item in benchmark}
        missing = [
            str(group.get("unique_id"))
            for group in groups
            if str(group.get("unique_id")) not in benchmark_by_id
        ]
        if missing:
            raise KeyError(f"Result unique_ids missing from MATH-500: {missing[:5]}")
        tokenizer = configure_decoder_only_tokenizer(
            AutoTokenizer.from_pretrained(
                args.base_model,
                revision=args.base_model_revision,
                trust_remote_code=True,
            )
        )
        ordered_items = [benchmark_by_id[str(group["unique_id"])] for group in groups]
        prompts = build_prompts(
            ordered_items,
            tokenizer,
            benchmark_name="math500",
            prompt_template=args.prompt_template,
        )
        prompt_token_ids = [
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for prompt in prompts
        ]
        prompt_source = "legacy_dataset_and_tokenizer_reconstruction"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(args.checkpoint), is_trainable=False)
    model.eval()

    frozen_ae = FrozenAE.load_frozen(args.frozen_ae)
    frozen_ae.to(device=device, dtype=torch.float32).eval()
    projection, mrn, head_state = _strict_load_heads(args.checkpoint, device)
    checkpoint_beta = float(head_state.get("cr_beta", args.beta))
    if abs(checkpoint_beta - args.beta) > 1e-12:
        raise ValueError(
            f"Requested beta={args.beta} does not match checkpoint cr_beta={checkpoint_beta}"
        )

    records: list[dict[str, Any]] = []
    pairwise_distance_groups: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    truncation_count = 0
    max_dropped_prompt_tokens = 0
    encoded_rollouts = 0
    encoded_mixed_rollouts = 0

    for group_index, (group, prompt_ids) in enumerate(
        zip(groups, prompt_token_ids, strict=True)
    ):
        samples = group["samples"]
        group_id = str(group["unique_id"])
        if args.pairwise_only:
            kind = None
            base_rows = []
        else:
            kind = _group_kind(samples)
            kind_counts[kind] += 1
            base_rows = [
                {
                    "group_id": group_id,
                    "unique_id": group_id,
                    "group_index": group_index,
                    "sample_idx": sample_idx,
                    "correct": bool(sample["correct"]),
                    "group_type": kind,
                    "n_correct": int(group["n_correct"]),
                    "level": int(group.get("level", 0)),
                    "subject": group.get("subject"),
                    "diagnostic_distance": None,
                    "deployed_distance": 0.0,
                }
                for sample_idx, sample in enumerate(samples)
            ]

        raw_rows = []
        projected_rows = []
        for sample in samples:
            completion_ids = [int(token) for token in sample["token_ids"]]
            full_ids, dropped = _prepare_sequence(
                prompt_ids,
                completion_ids,
                max_model_len=args.max_model_len,
            )
            if dropped:
                truncation_count += 1
                max_dropped_prompt_tokens = max(max_dropped_prompt_tokens, dropped)
            raw_latent, projected_latent = _project_completion(
                model=model,
                frozen_ae=frozen_ae,
                projection_head=projection,
                input_ids=full_ids,
                completion_len=len(completion_ids),
                num_layers_averaged=args.num_layers_averaged,
                device=device,
            )
            raw_rows.append(raw_latent)
            projected_rows.append(projected_latent)
            encoded_rollouts += 1
            if kind == "mixed":
                encoded_mixed_rollouts += 1

        raw_latents = torch.stack(raw_rows).to(device=device, dtype=torch.float32)
        projected = torch.stack(projected_rows).to(device=device, dtype=torch.float32)
        pairwise_matrices = _pairwise_distance_matrices(
            raw_latents=raw_latents,
            projected_latents=projected,
            mrn=mrn,
        )
        # This block intentionally contains identifiers and geometry only.  In
        # particular, it contains no correctness labels, gold answers, or
        # oracle-selected reference indices.
        pairwise_distance_groups.append(
            {
                "group_id": group_id,
                "unique_id": group_id,
                "group_index": group_index,
                "n_samples": len(samples),
                "sample_indices": list(range(len(samples))),
                "matrices": {
                    name: matrix.detach().cpu().tolist()
                    for name, matrix in pairwise_matrices.items()
                },
            }
        )

        if (group_index + 1) % 25 == 0:
            print(
                f"processed {group_index + 1}/{len(groups)} groups; "
                f"encoded_rollouts={encoded_rollouts}",
                flush=True,
            )

        if args.pairwise_only:
            continue

        if kind != "mixed":
            records.extend(base_rows)
            continue

        correct_mask = torch.as_tensor(
            [bool(sample["correct"]) for sample in samples],
            dtype=torch.bool,
            device=device,
        )
        correct_indices = torch.nonzero(correct_mask, as_tuple=False).flatten()
        with torch.inference_mode():
            for sample_idx, row in enumerate(base_rows):
                if row["correct"]:
                    other_indices = correct_indices[correct_indices != sample_idx]
                    if other_indices.numel():
                        distance = pairwise_matrices["mrn"][
                            sample_idx, other_indices
                        ].min()
                        row["diagnostic_distance"] = float(distance.item())
                    # The trainer never penalises a correct rollout.
                    row["deployed_distance"] = 0.0
                else:
                    distances = pairwise_matrices["mrn"][sample_idx, correct_indices]
                    distance = float(distances.min().item())
                    row["diagnostic_distance"] = distance
                    row["deployed_distance"] = distance
                records.append(row)

    output = {
        "schema_version": 2,
        "metadata": {
            "source_results": str(args.results),
            "source_results_sha256": sha256_file(args.results),
            "source_run_metadata": source_run_metadata,
            "generator_arm": (
                source_run_metadata.get("arm_label") if source_run_metadata else None
            ),
            "generator_checkpoint_sha256": (
                source_run_metadata.get("checkpoint_tree_sha256")
                if source_run_metadata
                else None
            ),
            "checkpoint": str(args.checkpoint),
            "checkpoint_tree_sha256": scorer_checkpoint_sha256,
            "scorer_checkpoint": str(args.checkpoint),
            "scorer_checkpoint_sha256": scorer_checkpoint_sha256,
            "scorer_checkpoint_role": "fixed historical G7 policy adapter plus projection/MRN heads used to teacher-force every arm's candidates in one representation coordinate system",
            "checkpoint_tree_hash_schema": "lavida.eval.provenance.sha256_tree.v1",
            "scorer_checkpoint_tree_summary": {
                key: scorer_checkpoint_inventory[key]
                for key in ("kind", "sha256", "file_count", "total_bytes")
            },
            "frozen_ae": str(args.frozen_ae),
            "frozen_ae_sha256": frozen_ae_sha256,
            "base_model": args.base_model,
            "base_model_revision": args.base_model_revision,
            "prompt_template": args.prompt_template,
            "prompt_token_source": prompt_source,
            "strict_provenance": args.strict_provenance,
            "protocol_file": str(args.protocol) if args.strict_provenance else None,
            "protocol_file_sha256": (
                sha256_file(args.protocol) if args.strict_provenance else None
            ),
            "protocol_lock_file": (
                str(args.protocol_lock) if args.strict_provenance else None
            ),
            "protocol_lock_file_sha256": (
                sha256_file(args.protocol_lock) if args.strict_provenance else None
            ),
            "max_model_len": args.max_model_len,
            "num_layers_averaged": args.num_layers_averaged,
            "beta": args.beta,
            "expected_group_size": args.expected_group_size,
            "expected_groups": args.expected_groups,
            "n_groups": len(groups),
            "n_rollouts": encoded_rollouts,
            "n_outcome_records": len(records),
            "legacy_records_mode": (
                "omitted_pairwise_only"
                if args.pairwise_only
                else "correctness_conditioned_records_included"
            ),
            "encoded_rollouts_all_groups": encoded_rollouts,
            "encoded_mixed_rollouts": (
                None if args.pairwise_only else encoded_mixed_rollouts
            ),
            "group_counts": None if args.pairwise_only else dict(kind_counts),
            "teacher_forcing_context_truncation": {
                "n_rollouts": truncation_count,
                "max_dropped_prompt_tokens": max_dropped_prompt_tokens,
                "rule": "left-truncated prompt context only; full completion preserved",
            },
            "wall_clock_seconds": time.time() - started,
            "hardware": torch.cuda.get_device_name(0),
            "post_hoc": not args.strict_provenance,
            "analysis_timing": (
                "prospective_reviewer_motivated_followup_under_locked_protocol"
                if args.strict_provenance
                else "post_hoc_diagnostic"
            ),
            "distance_semantics": {
                "diagnostic_distance": "nearest other correct rollout when defined",
                "deployed_distance": "zero for correct/degenerate rollouts; nearest correct distance for incorrect mixed-group rollouts",
                "generator_scope": "executed G7 reward semantics only when generator_arm is G7; a counterfactual G7-scorer diagnostic for A/D/G sources",
            },
            "pairwise_distance_semantics": {
                "field": "pairwise_distance_groups",
                "label_free_construction": True,
                "candidate_set": "all saved candidates in each prompt group",
                "mrn": "symmetric learned Mahalanobis distance between each pair of projected candidates; distinct from the outcome-gated asymmetric reward application",
                "projected_l2": "Euclidean distance between every pair of projected candidates",
                "raw_l2": "Euclidean distance between every pair of frozen-VAE posterior means",
                "diagonal": "set to exact zero by convention",
                "warning": "The MRN/projection heads were learned with outcome supervision; label-free here means current candidate correctness is not used to construct the matrices.",
            },
        },
        "records": records,
        "pairwise_distance_groups": pairwise_distance_groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
