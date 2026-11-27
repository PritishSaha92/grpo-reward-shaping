"""Audit whether a rollout-level shaping signal adds usable GRPO advantages.

The input is the per-rollout geometry JSON written by
``extract_g7_coverage_advantage_records.py``.  The analysis deliberately uses
the reward that the trainer actually deployed:

    r_i' = R_i - beta * d_i * 1[R_i = 0 and the prompt is mixed].

Correct rollouts receive no distance penalty, and all-correct/all-wrong groups
are skipped.  This differs from a symmetric ``R_i - beta*d_i`` thought
experiment and is important when interpreting advantage novelty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-4  # Matches LaViDATrainer._group_normalize_rewards.
ORDER_TOL = 1e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy-gradient",
        type=Path,
        default=None,
        help="Optional JSON from estimate_g7_policy_gradient_alignment.py.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override beta stored in the input metadata.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result


def load_groups(path: Path) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise TypeError(f"{path} must contain an object with a records list")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in payload["records"]:
        if not isinstance(row, dict):
            raise TypeError("Every records entry must be an object")
        group_id = str(row.get("group_id", row.get("unique_id", "")))
        if not group_id:
            raise ValueError("Every record needs group_id or unique_id")
        sample_idx = int(row["sample_idx"])
        key = (group_id, sample_idx)
        if key in seen:
            raise ValueError(f"Duplicate rollout key: {key}")
        seen.add(key)
        normalised = dict(row)
        normalised["group_id"] = group_id
        normalised["sample_idx"] = sample_idx
        normalised["correct"] = bool(row["correct"])
        grouped[group_id].append(normalised)

    groups = []
    for group_id in sorted(grouped):
        group = sorted(grouped[group_id], key=lambda row: row["sample_idx"])
        expected = list(range(len(group)))
        observed = [row["sample_idx"] for row in group]
        if observed != expected:
            raise ValueError(
                f"Group {group_id} has sample indices {observed}, expected {expected}"
            )
        groups.append(group)
    return payload, groups


def group_type(rewards: np.ndarray) -> str:
    n_correct = int(rewards.sum())
    if n_correct == 0:
        return "zero_correct"
    if n_correct == rewards.size:
        return "all_correct"
    return "mixed"


def group_normalise(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    centered = rewards - rewards.mean()
    return centered / (rewards.std(ddof=0) + EPS)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std(ddof=0) <= 0.0 or y.std(ddof=0) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(x, y) / denom)


def _orthogonal_energy_fraction(x: np.ndarray, y: np.ndarray) -> float:
    cosine = _cosine(x, y)
    if not math.isfinite(cosine):
        return math.nan
    return float(np.clip(1.0 - cosine**2, 0.0, 1.0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else math.nan


def _pairwise_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> tuple[float, int]:
    """Return AUROC and positive-negative pair count without external packages."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if not positive_scores.size or not negative_scores.size:
        return math.nan, 0
    differences = positive_scores[:, None] - negative_scores[None, :]
    wins = float(np.count_nonzero(differences > ORDER_TOL))
    ties = float(np.count_nonzero(np.abs(differences) <= ORDER_TOL))
    pairs = int(differences.size)
    return (wins + 0.5 * ties) / pairs, pairs


def _distance_for_row(row: dict[str, Any], kind: str) -> float:
    value = row.get("deployed_distance")
    if value is None:
        if kind == "mixed" and not row["correct"]:
            raise ValueError(
                "A mixed-group incorrect rollout is missing deployed_distance: "
                f"{row['group_id']} sample {row['sample_idx']}"
            )
        return 0.0
    return _as_float(value, label="deployed_distance")


def compute_core_metrics(
    groups: list[list[dict[str, Any]]],
    beta: float,
    *,
    include_order: bool = True,
    include_diagnostic: bool = True,
) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    all_binary_adv: list[np.ndarray] = []
    all_shaped_adv: list[np.ndarray] = []
    mixed_binary_adv: list[np.ndarray] = []
    mixed_shaped_adv: list[np.ndarray] = []
    mixed_distances: list[np.ndarray] = []
    mixed_rewards: list[np.ndarray] = []
    mixed_centered_distances: list[np.ndarray] = []
    mixed_centered_rewards: list[np.ndarray] = []
    saturated_residuals: list[np.ndarray] = []
    relative_delta_norms: list[float] = []
    diagnostic_positive_scores: list[float] = []
    diagnostic_negative_scores: list[float] = []
    conditional_auc_weighted_sum = 0.0
    conditional_auc_pairs = 0
    conditional_group_aucs: list[float] = []
    diagnostic_centered_distance: list[np.ndarray] = []
    diagnostic_centered_reward: list[np.ndarray] = []
    diagnostic_scorable = 0
    nonzero_bonus_rollouts = 0
    advantage_changed_rollouts = 0
    n_rollouts = 0

    pair_counts: Counter[str] = Counter()
    sign_counts: Counter[str] = Counter()

    for group in groups:
        rewards = np.asarray([float(row["correct"]) for row in group], dtype=np.float64)
        kind = group_type(rewards)
        distances = np.asarray(
            [_distance_for_row(row, kind) for row in group], dtype=np.float64
        )
        shaped_rewards = rewards - beta * distances
        binary_adv = group_normalise(rewards)
        shaped_adv = group_normalise(shaped_rewards)
        delta_adv = shaped_adv - binary_adv

        for reward, base_value, shaped_value in zip(
            rewards, binary_adv, shaped_adv, strict=True
        ):
            base_sign = 0 if abs(base_value) <= ORDER_TOL else int(np.sign(base_value))
            shaped_sign = (
                0 if abs(shaped_value) <= ORDER_TOL else int(np.sign(shaped_value))
            )
            sign_counts["rollouts"] += 1
            if kind == "mixed":
                sign_counts["mixed_rollouts"] += 1
                if not reward:
                    sign_counts["mixed_incorrect_rollouts"] += 1
            if base_sign and shaped_sign == -base_sign:
                sign_counts["strict_advantage_sign_flips"] += 1
                if kind == "mixed" and not reward:
                    sign_counts["mixed_incorrect_negative_to_positive"] += 1
            elif base_sign and shaped_sign == 0:
                sign_counts["advantage_sign_collapses"] += 1

        n_rollouts += len(group)
        diagnostic_scorable += sum(row.get("diagnostic_distance") is not None for row in group)
        nonzero_bonus_rollouts += int(np.count_nonzero(np.abs(beta * distances) > ORDER_TOL))
        advantage_changed_rollouts += int(np.count_nonzero(np.abs(delta_adv) > ORDER_TOL))
        all_binary_adv.append(binary_adv)
        all_shaped_adv.append(shaped_adv)

        base_norm = float(np.linalg.norm(binary_adv))
        delta_norm = float(np.linalg.norm(delta_adv))
        group_rows.append(
            {
                "type": kind,
                "n": len(group),
                "n_correct": int(rewards.sum()),
                "binary_adv_norm": base_norm,
                "shaped_adv_norm": float(np.linalg.norm(shaped_adv)),
                "relative_delta_norm": _safe_ratio(delta_norm, base_norm),
                "advantage_cosine": _cosine(binary_adv, shaped_adv),
                "novel_advantage_energy_fraction": _orthogonal_energy_fraction(
                    binary_adv, shaped_adv
                ),
                "outcome_gap_multiplier": (
                    1.0 + beta * float(distances[rewards == 0.0].mean())
                    if kind == "mixed"
                    else math.nan
                ),
            }
        )

        if kind == "mixed":
            mixed_binary_adv.append(binary_adv)
            mixed_shaped_adv.append(shaped_adv)
            mixed_distances.append(distances)
            mixed_rewards.append(rewards)
            d_centered = distances - distances.mean()
            r_centered = rewards - rewards.mean()
            mixed_centered_distances.append(d_centered)
            mixed_centered_rewards.append(r_centered)
            if base_norm > 0.0:
                relative_delta_norms.append(delta_norm / base_norm)

            residual = np.empty_like(distances)
            for outcome in (0.0, 1.0):
                mask = rewards == outcome
                residual[mask] = distances[mask] - distances[mask].mean()
            saturated_residuals.append(residual)

            if include_diagnostic:
                diagnostic_rows = [
                    (float(row["correct"]), -float(row["diagnostic_distance"]))
                    for row in group
                    if row.get("diagnostic_distance") is not None
                ]
                if diagnostic_rows:
                    diag_reward = np.asarray([row[0] for row in diagnostic_rows])
                    diag_score = np.asarray([row[1] for row in diagnostic_rows])
                    diagnostic_positive_scores.extend(
                        diag_score[diag_reward == 1.0].tolist()
                    )
                    diagnostic_negative_scores.extend(
                        diag_score[diag_reward == 0.0].tolist()
                    )
                    group_auc, group_pairs = _pairwise_auc(
                        diag_score[diag_reward == 1.0], diag_score[diag_reward == 0.0]
                    )
                    if group_pairs:
                        diagnostic_centered_distance.append(diag_score - diag_score.mean())
                        diagnostic_centered_reward.append(diag_reward - diag_reward.mean())
                        conditional_auc_weighted_sum += group_auc * group_pairs
                        conditional_auc_pairs += group_pairs
                        conditional_group_aucs.append(group_auc)

        if include_order:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    base_delta = rewards[i] - rewards[j]
                    shaped_delta = shaped_rewards[i] - shaped_rewards[j]
                    base_sign = 0 if abs(base_delta) <= ORDER_TOL else int(np.sign(base_delta))
                    shaped_sign = (
                        0 if abs(shaped_delta) <= ORDER_TOL else int(np.sign(shaped_delta))
                    )
                    pair_counts["all_pairs"] += 1
                    if base_sign:
                        pair_counts["cross_outcome_pairs"] += 1
                        if shaped_sign == -base_sign:
                            pair_counts["strict_reversals"] += 1
                        elif shaped_sign == 0:
                            pair_counts["cross_outcome_collapses"] += 1
                    else:
                        pair_counts["binary_tied_pairs"] += 1
                        if not rewards[i] and not rewards[j]:
                            pair_counts["incorrect_tied_pairs"] += 1
                            if kind == "mixed":
                                pair_counts["mixed_incorrect_tied_pairs"] += 1
                        else:
                            pair_counts["correct_tied_pairs"] += 1
                        if shaped_sign:
                            pair_counts["ties_broken"] += 1
                            if not rewards[i] and not rewards[j]:
                                pair_counts["incorrect_ties_broken"] += 1
                                if kind == "mixed":
                                    pair_counts["mixed_incorrect_ties_broken"] += 1
                            else:
                                pair_counts["correct_ties_broken"] += 1

    flat_binary = np.concatenate(all_binary_adv)
    flat_shaped = np.concatenate(all_shaped_adv)
    flat_mixed_binary = np.concatenate(mixed_binary_adv) if mixed_binary_adv else np.array([])
    flat_mixed_shaped = np.concatenate(mixed_shaped_adv) if mixed_shaped_adv else np.array([])
    flat_distances = np.concatenate(mixed_distances) if mixed_distances else np.array([])
    flat_rewards = np.concatenate(mixed_rewards) if mixed_rewards else np.array([])
    flat_dc = (
        np.concatenate(mixed_centered_distances) if mixed_centered_distances else np.array([])
    )
    flat_rc = np.concatenate(mixed_centered_rewards) if mixed_centered_rewards else np.array([])
    flat_saturated_residual = (
        np.concatenate(saturated_residuals) if saturated_residuals else np.array([])
    )
    pooled_diagnostic_auc, pooled_diagnostic_pairs = (
        _pairwise_auc(
            np.asarray(diagnostic_positive_scores),
            np.asarray(diagnostic_negative_scores),
        )
        if include_diagnostic
        else (math.nan, 0)
    )
    flat_diag_centered_distance = (
        np.concatenate(diagnostic_centered_distance)
        if diagnostic_centered_distance
        else np.array([])
    )
    flat_diag_centered_reward = (
        np.concatenate(diagnostic_centered_reward)
        if diagnostic_centered_reward
        else np.array([])
    )

    type_counts = Counter(row["type"] for row in group_rows)
    type_rollouts = Counter()
    for row in group_rows:
        type_rollouts[row["type"]] += int(row["n"])

    # Global correctness-only regression: d = intercept + slope * R + epsilon.
    if flat_distances.size:
        design = np.column_stack([np.ones_like(flat_rewards), flat_rewards])
        coef, *_ = np.linalg.lstsq(design, flat_distances, rcond=None)
        global_residual = flat_distances - design @ coef
        global_residual_fraction = _safe_ratio(
            float(np.var(global_residual, ddof=0)),
            float(np.var(flat_distances, ddof=0)),
        )
    else:
        coef = np.asarray([math.nan, math.nan])
        global_residual_fraction = math.nan

    # Prompt centering removes c_p; this estimates one common correctness slope.
    centered_denom = float(np.dot(flat_dc, flat_dc))
    correctness_denom = float(np.dot(flat_rc, flat_rc))
    centered_slope = (
        float(np.dot(flat_rc, flat_dc) / correctness_denom)
        if correctness_denom > 0.0
        else math.nan
    )
    centered_residual = flat_dc - centered_slope * flat_rc
    centered_residual_fraction = _safe_ratio(
        float(np.dot(centered_residual, centered_residual)), centered_denom
    )

    # Saturated prompt x outcome means leave only within-outcome ordering.
    saturated_residual_fraction = _safe_ratio(
        float(np.dot(flat_saturated_residual, flat_saturated_residual)),
        centered_denom,
    )

    mixed_group_rows = [row for row in group_rows if row["type"] == "mixed"]
    result: dict[str, Any] = {
        "coverage": {
            "n_groups": len(groups),
            "n_rollouts": n_rollouts,
            "groups": dict(type_counts),
            "group_fractions": {
                key: value / len(groups) for key, value in sorted(type_counts.items())
            },
            "rollouts": dict(type_rollouts),
            "rollout_fractions": {
                key: value / n_rollouts for key, value in sorted(type_rollouts.items())
            },
            "zero_correct_group_signal_coverage": 0.0,
            "mixed_group_signal_coverage": (
                1.0 if type_counts.get("mixed", 0) else math.nan
            ),
            "all_correct_group_signal_coverage": 0.0,
            "nonzero_bonus_rollouts": nonzero_bonus_rollouts,
            "nonzero_bonus_rollout_fraction": nonzero_bonus_rollouts / n_rollouts,
            "diagnostic_distance_scorable_rollouts": diagnostic_scorable,
            "diagnostic_distance_scorable_fraction": diagnostic_scorable / n_rollouts,
        },
        "advantage": {
            "beta": beta,
            "pearson_shaped_vs_binary_all_rollouts": _pearson(
                flat_binary, flat_shaped
            ),
            "pearson_shaped_vs_binary_mixed_rollouts": _pearson(
                flat_mixed_binary, flat_mixed_shaped
            ),
            "cosine_shaped_vs_binary_all_rollouts": _cosine(
                flat_binary, flat_shaped
            ),
            "cosine_shaped_vs_binary_mixed_rollouts": _cosine(
                flat_mixed_binary, flat_mixed_shaped
            ),
            "norm_ratio_shaped_to_binary_all_rollouts": _safe_ratio(
                float(np.linalg.norm(flat_shaped)), float(np.linalg.norm(flat_binary))
            ),
            "relative_delta_norm_all_rollouts": _safe_ratio(
                float(np.linalg.norm(flat_shaped - flat_binary)),
                float(np.linalg.norm(flat_binary)),
            ),
            "novel_advantage_energy_fraction_all_rollouts": (
                _orthogonal_energy_fraction(flat_binary, flat_shaped)
            ),
            "mixed_group_relative_delta_norm_mean": (
                float(np.mean(relative_delta_norms)) if relative_delta_norms else math.nan
            ),
            "mixed_group_relative_delta_norm_median": (
                float(np.median(relative_delta_norms)) if relative_delta_norms else math.nan
            ),
            "advantage_changed_rollouts": advantage_changed_rollouts,
            "advantage_changed_rollout_fraction": advantage_changed_rollouts / n_rollouts,
        },
        "novelty": {
            "distance_correctness_regression_intercept": float(coef[0]),
            "distance_correctness_regression_slope": float(coef[1]),
            "residual_variance_fraction_after_global_correctness_regression": (
                global_residual_fraction
            ),
            "prompt_centered_correctness_slope": centered_slope,
            "residual_variance_fraction_after_prompt_centering_and_correctness": (
                centered_residual_fraction
            ),
            "within_outcome_variance_fraction_after_prompt_x_outcome_means": (
                saturated_residual_fraction
            ),
        },
        "advantage_sign": {
            "strict_advantage_sign_flips": int(
                sign_counts["strict_advantage_sign_flips"]
            ),
            "strict_advantage_sign_flip_rate_all_rollouts": _safe_ratio(
                sign_counts["strict_advantage_sign_flips"], sign_counts["rollouts"]
            ),
            "mixed_incorrect_negative_to_positive": int(
                sign_counts["mixed_incorrect_negative_to_positive"]
            ),
            "mixed_incorrect_negative_to_positive_rate": _safe_ratio(
                sign_counts["mixed_incorrect_negative_to_positive"],
                sign_counts["mixed_incorrect_rollouts"],
            ),
            "advantage_sign_collapses": int(sign_counts["advantage_sign_collapses"]),
        },
    }

    if include_diagnostic:
        result["diagnostic_geometry"] = {
            "pooled_auroc_neg_distance": pooled_diagnostic_auc,
            "pooled_positive_negative_pairs": pooled_diagnostic_pairs,
            "within_prompt_pair_weighted_auroc_neg_distance": _safe_ratio(
                conditional_auc_weighted_sum, conditional_auc_pairs
            ),
            "within_prompt_positive_negative_pairs": conditional_auc_pairs,
            "within_prompt_groups_scorable": len(conditional_group_aucs),
            "within_prompt_macro_auroc_mean": (
                float(np.mean(conditional_group_aucs))
                if conditional_group_aucs
                else math.nan
            ),
            "prompt_centered_score_correctness_pearson": _pearson(
                flat_diag_centered_distance, flat_diag_centered_reward
            ),
            "selection_note": (
                "Correct rollouts in lone-correct groups have undefined diagnostic distance; "
                "within-prompt AUROC excludes those groups. Deployed-distance analysis does not."
            ),
        }

    if include_order:
        pair_names = (
            "all_pairs",
            "cross_outcome_pairs",
            "strict_reversals",
            "cross_outcome_collapses",
            "binary_tied_pairs",
            "incorrect_tied_pairs",
            "mixed_incorrect_tied_pairs",
            "correct_tied_pairs",
            "ties_broken",
            "incorrect_ties_broken",
            "mixed_incorrect_ties_broken",
            "correct_ties_broken",
        )
        result["ordering"] = {
            **{name: int(pair_counts[name]) for name in pair_names},
            "strict_cross_outcome_reversal_rate": _safe_ratio(
                pair_counts["strict_reversals"], pair_counts["cross_outcome_pairs"]
            ),
            "binary_tie_break_rate": _safe_ratio(
                pair_counts["ties_broken"], pair_counts["binary_tied_pairs"]
            ),
            "incorrect_tie_break_rate": _safe_ratio(
                pair_counts["incorrect_ties_broken"], pair_counts["incorrect_tied_pairs"]
            ),
            "mixed_incorrect_tie_break_rate": _safe_ratio(
                pair_counts["mixed_incorrect_ties_broken"],
                pair_counts["mixed_incorrect_tied_pairs"],
            ),
            "correct_tie_break_rate": _safe_ratio(
                pair_counts["correct_ties_broken"], pair_counts["correct_tied_pairs"]
            ),
        }

    if mixed_group_rows:
        result["advantage"]["mixed_group_advantage_cosine_mean"] = float(
            np.mean([row["advantage_cosine"] for row in mixed_group_rows])
        )
        result["advantage"]["mixed_group_advantage_cosine_median"] = float(
            np.median([row["advantage_cosine"] for row in mixed_group_rows])
        )
        result["advantage"]["mixed_group_novel_energy_fraction_mean"] = float(
            np.mean(
                [row["novel_advantage_energy_fraction"] for row in mixed_group_rows]
            )
        )
        result["advantage"]["mixed_group_novel_energy_fraction_median"] = float(
            np.median(
                [row["novel_advantage_energy_fraction"] for row in mixed_group_rows]
            )
        )
        result["advantage"]["mixed_group_outcome_gap_multiplier_mean"] = float(
            np.mean([row["outcome_gap_multiplier"] for row in mixed_group_rows])
        )
        result["advantage"]["mixed_group_outcome_gap_multiplier_median"] = float(
            np.median([row["outcome_gap_multiplier"] for row in mixed_group_rows])
        )
    return result


BOOTSTRAP_PATHS = {
    "mixed_group_fraction": ("coverage", "group_fractions", "mixed"),
    "nonzero_bonus_rollout_fraction": (
        "coverage",
        "nonzero_bonus_rollout_fraction",
    ),
    "advantage_pearson_mixed": (
        "advantage",
        "pearson_shaped_vs_binary_mixed_rollouts",
    ),
    "advantage_cosine_mixed": (
        "advantage",
        "cosine_shaped_vs_binary_mixed_rollouts",
    ),
    "advantage_norm_ratio_all": (
        "advantage",
        "norm_ratio_shaped_to_binary_all_rollouts",
    ),
    "advantage_relative_delta_norm_all": (
        "advantage",
        "relative_delta_norm_all_rollouts",
    ),
    "novel_advantage_energy_fraction_all": (
        "advantage",
        "novel_advantage_energy_fraction_all_rollouts",
    ),
    "global_correctness_residual_fraction": (
        "novelty",
        "residual_variance_fraction_after_global_correctness_regression",
    ),
    "prompt_centered_correctness_residual_fraction": (
        "novelty",
        "residual_variance_fraction_after_prompt_centering_and_correctness",
    ),
    "within_outcome_variance_fraction": (
        "novelty",
        "within_outcome_variance_fraction_after_prompt_x_outcome_means",
    ),
    "within_prompt_diagnostic_auroc": (
        "diagnostic_geometry",
        "within_prompt_pair_weighted_auroc_neg_distance",
    ),
}


def _get_path(mapping: dict[str, Any], path: Iterable[str]) -> float:
    value: Any = mapping
    for key in path:
        value = value[key]
    return float(value)


def bootstrap_intervals(
    groups: list[list[dict[str, Any]]],
    beta: float,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples <= 0:
        return {"resamples": 0, "seed": seed, "intervals": {}}

    # Every reported bootstrap scalar has an additive per-prompt sufficient
    # statistic.  Summing those statistics in vectorised chunks is exactly the
    # same prompt bootstrap as rebuilding rollout arrays, but makes 20k
    # resamples practical on CPU.
    columns = {
        name: index
        for index, name in enumerate(
            (
                "mixed",
                "nonzero_bonus",
                "n_rollouts",
                "binary_sq",
                "shaped_sq",
                "cross",
                "delta_sq",
                "mixed_n",
                "sum_d",
                "sum_r",
                "sum_d2",
                "sum_r2",
                "sum_dr",
                "centered_d2",
                "centered_r2",
                "centered_dr",
                "saturated_residual2",
                "diagnostic_pair_wins",
                "diagnostic_pairs",
            )
        )
    }
    per_group = np.zeros((len(groups), len(columns)), dtype=np.float64)
    for group_index, group in enumerate(groups):
        rewards = np.asarray([float(row["correct"]) for row in group], dtype=np.float64)
        kind = group_type(rewards)
        distances = np.asarray(
            [_distance_for_row(row, kind) for row in group], dtype=np.float64
        )
        binary = group_normalise(rewards)
        shaped = group_normalise(rewards - beta * distances)
        delta = shaped - binary
        row = per_group[group_index]
        row[columns["mixed"]] = float(kind == "mixed")
        row[columns["nonzero_bonus"]] = np.count_nonzero(
            np.abs(beta * distances) > ORDER_TOL
        )
        row[columns["n_rollouts"]] = len(group)
        row[columns["binary_sq"]] = np.dot(binary, binary)
        row[columns["shaped_sq"]] = np.dot(shaped, shaped)
        row[columns["cross"]] = np.dot(binary, shaped)
        row[columns["delta_sq"]] = np.dot(delta, delta)
        if kind == "mixed":
            row[columns["mixed_n"]] = len(group)
            row[columns["sum_d"]] = distances.sum()
            row[columns["sum_r"]] = rewards.sum()
            row[columns["sum_d2"]] = np.dot(distances, distances)
            row[columns["sum_r2"]] = np.dot(rewards, rewards)
            row[columns["sum_dr"]] = np.dot(distances, rewards)
            dc = distances - distances.mean()
            rc = rewards - rewards.mean()
            row[columns["centered_d2"]] = np.dot(dc, dc)
            row[columns["centered_r2"]] = np.dot(rc, rc)
            row[columns["centered_dr"]] = np.dot(dc, rc)
            saturated = np.empty_like(distances)
            for outcome in (0.0, 1.0):
                mask = rewards == outcome
                saturated[mask] = distances[mask] - distances[mask].mean()
            row[columns["saturated_residual2"]] = np.dot(saturated, saturated)
            diagnostic = [
                (float(item["correct"]), -float(item["diagnostic_distance"]))
                for item in group
                if item.get("diagnostic_distance") is not None
            ]
            if diagnostic:
                diag_reward = np.asarray([item[0] for item in diagnostic])
                diag_score = np.asarray([item[1] for item in diagnostic])
                diag_auc, diag_pairs = _pairwise_auc(
                    diag_score[diag_reward == 1.0], diag_score[diag_reward == 0.0]
                )
                if diag_pairs:
                    row[columns["diagnostic_pair_wins"]] = diag_auc * diag_pairs
                    row[columns["diagnostic_pairs"]] = diag_pairs

    rng = np.random.default_rng(seed)
    values: dict[str, list[np.ndarray]] = {name: [] for name in BOOTSTRAP_PATHS}
    n_groups = len(groups)
    chunk_size = 500
    for start in range(0, resamples, chunk_size):
        batch = min(chunk_size, resamples - start)
        indices = rng.integers(0, n_groups, size=(batch, n_groups))
        sums = per_group[indices].sum(axis=1)

        def col(name: str) -> np.ndarray:
            return sums[:, columns[name]]

        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = col("cross") / np.sqrt(col("binary_sq") * col("shaped_sq"))
            norm_ratio = np.sqrt(col("shaped_sq") / col("binary_sq"))
            relative_delta = np.sqrt(col("delta_sq") / col("binary_sq"))
            novelty_energy = np.clip(1.0 - cosine**2, 0.0, 1.0)

            mixed_n = col("mixed_n")
            sst_d = col("sum_d2") - col("sum_d") ** 2 / mixed_n
            sst_r = col("sum_r2") - col("sum_r") ** 2 / mixed_n
            cross_centered = col("sum_dr") - col("sum_d") * col("sum_r") / mixed_n
            global_residual = (sst_d - cross_centered**2 / sst_r) / sst_d
            centered_residual = (
                col("centered_d2")
                - col("centered_dr") ** 2 / col("centered_r2")
            ) / col("centered_d2")
            saturated_fraction = col("saturated_residual2") / col("centered_d2")
            within_prompt_auc = col("diagnostic_pair_wins") / col("diagnostic_pairs")

        values["mixed_group_fraction"].append(col("mixed") / n_groups)
        values["nonzero_bonus_rollout_fraction"].append(
            col("nonzero_bonus") / col("n_rollouts")
        )
        values["advantage_pearson_mixed"].append(cosine)
        values["advantage_cosine_mixed"].append(cosine)
        values["advantage_norm_ratio_all"].append(norm_ratio)
        values["advantage_relative_delta_norm_all"].append(relative_delta)
        values["novel_advantage_energy_fraction_all"].append(novelty_energy)
        values["global_correctness_residual_fraction"].append(global_residual)
        values["prompt_centered_correctness_residual_fraction"].append(
            centered_residual
        )
        values["within_outcome_variance_fraction"].append(saturated_fraction)
        values["within_prompt_diagnostic_auroc"].append(within_prompt_auc)

    intervals = {}
    for name, raw in values.items():
        array = np.concatenate(raw).astype(np.float64, copy=False)
        array = array[np.isfinite(array)]
        intervals[name] = (
            [float(x) for x in np.percentile(array, [2.5, 97.5])]
            if array.size
            else [math.nan, math.nan]
        )
    return {"resamples": resamples, "seed": seed, "intervals": intervals}


def main() -> None:
    args = parse_args()
    payload, groups = load_groups(args.records)
    beta = args.beta
    if beta is None:
        beta = payload.get("metadata", {}).get("beta")
    if beta is None:
        raise ValueError("beta is missing; pass --beta or store metadata.beta")
    beta = _as_float(beta, label="beta")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    metrics = compute_core_metrics(groups, beta)
    metrics["bootstrap"] = bootstrap_intervals(
        groups,
        beta,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    metrics["definitions"] = {
        "advantage.novel_advantage_energy_fraction_all_rollouts": (
            "1-cosine^2 for concatenated post-hoc re-decode normalized advantage "
            "vectors; not an average group value or training-time estimate"
        ),
        "advantage.mixed_group_novel_energy_fraction_median": (
            "median of per-mixed-group 1-cosine^2 values"
        ),
        "diagnostic_geometry.pooled_auroc_neg_distance": (
            "oracle-conditioned correct-versus-incorrect discrimination on "
            "selected/scorable rollouts; not within-incorrect ordering"
        ),
        "diagnostic_geometry.within_prompt_pair_weighted_auroc_neg_distance": (
            "pair-weighted correct-versus-incorrect discrimination within eligible "
            "prompts; not within-incorrect ordering"
        ),
    }
    metrics["provenance"] = {
        "records": str(args.records),
        "records_sha256": file_sha256(args.records),
        "reward_rule": "R - beta * deployed_distance; deployed_distance is zero for correct and degenerate-group rollouts",
        "normalization": "per-prompt population std plus 1e-4",
        "post_hoc": True,
    }
    if "policy_gradient" in payload:
        metrics["policy_gradient"] = payload["policy_gradient"]
    if args.policy_gradient is not None:
        gradient_payload = json.loads(args.policy_gradient.read_text(encoding="utf-8"))
        if not isinstance(gradient_payload, dict) or "policy_gradient" not in gradient_payload:
            raise TypeError(
                f"{args.policy_gradient} must contain a policy_gradient object"
            )
        metrics["policy_gradient"] = gradient_payload["policy_gradient"]
        metrics["provenance"]["policy_gradient"] = str(args.policy_gradient)
        metrics["provenance"]["policy_gradient_sha256"] = file_sha256(
            args.policy_gradient
        )
        metrics["provenance"]["policy_gradient_imported_not_recomputed"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
