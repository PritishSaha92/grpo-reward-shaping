"""Recover exact G7 coverage and aligned-gap statistics from W&B history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def main() -> None:
    args = parse_args()
    with args.history.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No history rows in {args.history}")

    parsed = []
    for row in rows:
        reward = float(row["train/reward"])
        bonus_mean = float(row["train/cr_reward_bonus_mean"])
        skip_rate = float(row["train/cr_head_aux_skip_rate"])
        head_loss = float(row["train/loss/cr_head_aux"])
        offdiag = float(row["train/mrn_M_offdiag_norm"])
        if abs(reward) <= 1e-12:
            kind = "zero_correct"
        elif abs(reward - 1.0) <= 1e-12:
            kind = "all_correct"
        else:
            kind = "mixed"
        conditional_penalty = (
            -bonus_mean / (1.0 - reward) if kind == "mixed" else math.nan
        )
        parsed.append(
            {
                "reward": reward,
                "bonus_mean": bonus_mean,
                "skip_rate": skip_rate,
                "head_loss": head_loss,
                "offdiag": offdiag,
                "kind": kind,
                # beta * mean(d | incorrect, mixed).  This is the outcome-gap
                # amplification beyond the original unit binary gap.
                "conditional_penalty": conditional_penalty,
            }
        )

    counts = Counter(row["kind"] for row in parsed)
    mixed = [row for row in parsed if row["kind"] == "mixed"]
    nonzero_bonus_rollouts = int(
        round(
            sum(
                args.group_size * (1.0 - row["reward"])
                for row in mixed
            )
        )
    )
    total_rollouts = len(parsed) * args.group_size
    observed_skips = sum(row["skip_rate"] for row in parsed)
    expected_skips = counts["zero_correct"] + counts["all_correct"]
    if abs(observed_skips - expected_skips) > 1e-6:
        raise AssertionError(
            f"skip-rate sum {observed_skips} does not match degenerate groups {expected_skips}"
        )

    def window_summary(window: list[dict]) -> dict:
        active = [row for row in window if row["kind"] == "mixed"]
        penalties = [row["conditional_penalty"] for row in active]
        return {
            "n_groups": len(window),
            "n_mixed": len(active),
            "mean_reward": _mean([row["reward"] for row in window]),
            "mean_logged_bonus": _mean([row["bonus_mean"] for row in window]),
            "mean_beta_distance_given_incorrect_mixed": _mean(penalties),
            "median_beta_distance_given_incorrect_mixed": _median(penalties),
            "mean_active_head_loss": _mean([row["head_loss"] for row in active]),
        }

    penalties = [row["conditional_penalty"] for row in mixed]
    output = {
        "coverage": {
            "n_training_groups": len(parsed),
            "group_size": args.group_size,
            "n_training_rollouts": total_rollouts,
            "groups": dict(counts),
            "group_fractions": {
                key: value / len(parsed) for key, value in sorted(counts.items())
            },
            "nonzero_bonus_rollouts": nonzero_bonus_rollouts,
            "nonzero_bonus_rollout_fraction": nonzero_bonus_rollouts / total_rollouts,
            "zero_correct_rollouts_with_nonzero_bonus": 0,
            "frontier_skip_groups": int(round(observed_skips)),
        },
        "aligned_gap": {
            "mean_beta_distance_given_incorrect_mixed": _mean(penalties),
            "median_beta_distance_given_incorrect_mixed": _median(penalties),
            "mean_outcome_gap_multiplier": 1.0 + _mean(penalties),
            "median_outcome_gap_multiplier": 1.0 + _median(penalties),
        },
        "trainability": {
            "first_mrn_offdiag_norm": parsed[0]["offdiag"],
            "last_mrn_offdiag_norm": parsed[-1]["offdiag"],
            "active_group_head_loss_mean": _mean(
                [row["head_loss"] for row in mixed]
            ),
        },
        "first_100": window_summary(parsed[:100]),
        "last_100": window_summary(parsed[-100:]),
        "provenance": {
            "history": str(args.history),
            "history_sha256": file_sha256(args.history),
            "beta": args.beta,
            "reward_rule": "R - beta*(1-R)*distance on mixed groups only",
            "conditional_penalty_identity": "-logged_bonus_mean/(1-reward_mean)",
            "scope_warning": (
                "Group-level W&B rows do not contain rollout-level R, d, A, or "
                "A-prime and cannot reconstruct training-time advantage novelty."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
