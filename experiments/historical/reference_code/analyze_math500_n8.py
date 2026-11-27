"""Summarize n-sample MATH-500 evals and run paired significance tests.

Usage:
    python scripts/analyze_math500_n8.py \
        --root eval_results/math500_n8 \
        --compare baseline-math500-n8 varA-seed0-math500-n8 \
        --compare baseline-math500-n8 varB-seed1-step500-math500-n8

The script expects each run directory to contain `math500_pass8_results.json`.
It reports:
  - pass@8 (any-correct)
  - mean correctness across the 8 samples per problem
  - paired bootstrap CI for mean-correctness deltas
  - exact McNemar test on pass@8 correctness
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SUBSET_SPECS = {
    "L1": lambda row: row.get("level") == 1,
    "L2": lambda row: row.get("level") == 2,
    "L3": lambda row: row.get("level") == 3,
    "L4": lambda row: row.get("level") == 4,
    "L5": lambda row: row.get("level") == 5,
    "L1_3": lambda row: row.get("level") in {1, 2, 3},
    "L4_5": lambda row: row.get("level") in {4, 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("eval_results/math500_n8"),
        help="Directory containing run subdirectories with math500_pass8_results.json",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional explicit run mapping to a math500_pass8_results.json file",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=[],
        help="Run paired bootstrap + McNemar on RIGHT minus LEFT",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of paired bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap resampling",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default=None,
        help="Run label to use as the baseline for subset-vs-baseline comparisons",
    )
    parser.add_argument(
        "--subset-vs-baseline",
        action="append",
        choices=sorted(SUBSET_SPECS.keys()),
        default=[],
        help="Subset label to compare for all non-baseline runs against --baseline-label",
    )
    return parser.parse_args()


def _round_pct(value: float) -> float:
    return round(value * 100.0, 2)


def _mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def _problem_key(problem: dict, idx: int) -> str:
    unique_id = problem.get("unique_id")
    if unique_id is not None:
        return str(unique_id)
    return f"idx:{idx}"


def load_run(path: Path) -> dict:
    payload = json.loads(path.read_text())
    problems = payload["per_problem"]
    ordered = []
    means = []
    any_correct = []
    sample_counts = []
    for idx, item in enumerate(problems):
        n_samples = len(item["samples"])
        mean_correct = item["n_correct"] / max(n_samples, 1)
        key = _problem_key(item, idx)
        ordered.append(
            {
                "key": key,
                "level": item.get("level"),
                "mean_correct": mean_correct,
                "any_correct": bool(item["any_correct"]),
            }
        )
        means.append(mean_correct)
        any_correct.append(bool(item["any_correct"]))
        sample_counts.append(n_samples)

    mean_correctness, problem_se = _mean_and_se(means)
    total_samples = sum(sample_counts)
    binomial_se = math.sqrt(
        mean_correctness * (1.0 - mean_correctness) / max(total_samples, 1)
    )
    pass_at_8 = sum(any_correct) / max(len(any_correct), 1)

    return {
        "path": path,
        "payload": payload,
        "ordered": ordered,
        "means": means,
        "any_correct": any_correct,
        "n_problems": len(problems),
        "samples_per_problem": sample_counts[0] if sample_counts and len(set(sample_counts)) == 1 else None,
        "mean_correctness": mean_correctness,
        "problem_se": problem_se,
        "binomial_se": binomial_se,
        "pass_at_8": pass_at_8,
    }


def discover_runs(args: argparse.Namespace) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    if args.run:
        for spec in args.run:
            label, raw_path = spec.split("=", 1)
            runs[label] = load_run(Path(raw_path))
        return runs

    for path in sorted(args.root.glob("*/math500_pass8_results.json")):
        runs[path.parent.name] = load_run(path)
    return runs


def align_runs(left: dict, right: dict) -> tuple[list[float], list[float], list[bool], list[bool]]:
    left_keys = [row["key"] for row in left["ordered"]]
    right_keys = [row["key"] for row in right["ordered"]]
    if left_keys != right_keys:
        raise ValueError(
            "Run problem order/keys do not match; cannot perform paired comparison:\n"
            f"  left={left['path']}\n"
            f"  right={right['path']}"
        )
    return (
        [row["mean_correct"] for row in left["ordered"]],
        [row["mean_correct"] for row in right["ordered"]],
        [row["any_correct"] for row in left["ordered"]],
        [row["any_correct"] for row in right["ordered"]],
    )


def paired_bootstrap(
    left: list[float],
    right: list[float],
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equal-length vectors")
    n = len(left)
    diffs = [r - l for l, r in zip(left, right)]
    observed = sum(diffs) / max(n, 1)

    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot.append(total / n)
    boot.sort()

    lo_idx = int(0.025 * (n_boot - 1))
    hi_idx = int(0.975 * (n_boot - 1))
    non_positive = sum(x <= 0.0 for x in boot) / n_boot
    non_negative = sum(x >= 0.0 for x in boot) / n_boot
    p_two_sided = min(1.0, 2.0 * min(non_positive, non_negative))
    return {
        "delta": observed,
        "ci_low": boot[lo_idx],
        "ci_high": boot[hi_idx],
        "p_two_sided": p_two_sided,
    }


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("McNemar requires equal-length vectors")
    left_only = sum(1 for l, r in zip(left, right) if l and not r)
    right_only = sum(1 for l, r in zip(left, right) if r and not l)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, i) for i in range(0, min(left_only, right_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def filter_rows(run: dict, subset_label: str) -> list[dict]:
    predicate = SUBSET_SPECS[subset_label]
    rows = [row for row in run["ordered"] if predicate(row)]
    if not rows:
        raise ValueError(f"subset {subset_label} produced no rows for {run['path']}")
    return rows


def summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    means = [row["mean_correct"] for row in rows]
    any_correct = [bool(row["any_correct"]) for row in rows]
    mean_correctness, problem_se = _mean_and_se(means)
    total_samples = len(rows) * 8
    binomial_se = math.sqrt(
        mean_correctness * (1.0 - mean_correctness) / max(total_samples, 1)
    )
    pass_at_8 = sum(any_correct) / max(len(any_correct), 1)
    return {
        "n": len(rows),
        "mean_correctness": mean_correctness,
        "problem_se": problem_se,
        "binomial_se": binomial_se,
        "pass_at_8": pass_at_8,
    }


def print_run_table(runs: dict[str, dict]) -> None:
    print("\nRun Summary")
    print(
        f"{'run':34s} {'mean_corr':>10s} {'prob_SE':>9s} {'binom_SE':>10s} {'pass@8':>8s}"
    )
    for label, run in runs.items():
        print(
            f"{label:34s} "
            f"{_round_pct(run['mean_correctness']):9.2f}% "
            f"{_round_pct(run['problem_se']):8.2f}pp "
            f"{_round_pct(run['binomial_se']):9.2f}pp "
            f"{_round_pct(run['pass_at_8']):7.2f}%"
        )


def print_level_stratified_tables(runs: dict[str, dict]) -> None:
    subset_order = ["L1", "L2", "L3", "L4", "L5", "L1_3", "L4_5"]
    print("\nLevel-Stratified Summary")
    for subset_label in subset_order:
        print(f"\n{subset_label}")
        print(
            f"{'run':34s} {'n':>5s} {'mean_corr':>10s} {'prob_SE':>9s} {'binom_SE':>10s} {'pass@8':>8s}"
        )
        for label, run in runs.items():
            stats = summarize_rows(filter_rows(run, subset_label))
            print(
                f"{label:34s} "
                f"{stats['n']:5d} "
                f"{_round_pct(stats['mean_correctness']):9.2f}% "
                f"{_round_pct(stats['problem_se']):8.2f}pp "
                f"{_round_pct(stats['binomial_se']):9.2f}pp "
                f"{_round_pct(stats['pass_at_8']):7.2f}%"
            )


def print_comparisons(
    runs: dict[str, dict],
    comparisons: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> None:
    if not comparisons:
        return
    print("\nPaired Comparisons")
    for left_label, right_label in comparisons:
        left = runs[left_label]
        right = runs[right_label]
        left_means, right_means, left_any, right_any = align_runs(left, right)
        boot = paired_bootstrap(
            left_means, right_means, n_boot=n_boot, seed=seed
        )
        mcnemar = exact_mcnemar(left_any, right_any)
        pass_delta = right["pass_at_8"] - left["pass_at_8"]
        print(f"\n{left_label} -> {right_label}")
        print(
            "  Delta mean correctness (right-left): "
            f"{_round_pct(boot['delta']):+.2f}pp"
        )
        print(
            "  95% paired-bootstrap CI: "
            f"[{_round_pct(boot['ci_low']):+.2f}, {_round_pct(boot['ci_high']):+.2f}]pp"
        )
        print(
            "  Paired-bootstrap two-sided p: "
            f"{boot['p_two_sided']:.4f}"
        )
        print(f"  Delta pass@8 (right-left): {_round_pct(pass_delta):+.2f}pp")
        print(
            "  Exact McNemar p: "
            f"{mcnemar['p_value']:.4f}"
            f"  [left_only={mcnemar['left_only']}, right_only={mcnemar['right_only']}]"
        )


def print_subset_vs_baseline(
    runs: dict[str, dict],
    baseline_label: str,
    subset_labels: list[str],
    n_boot: int,
    seed: int,
) -> None:
    if not subset_labels:
        return
    if baseline_label not in runs:
        raise ValueError(f"baseline label {baseline_label!r} not found in runs")

    baseline = runs[baseline_label]
    for subset_label in subset_labels:
        baseline_rows = filter_rows(baseline, subset_label)
        baseline_means = [row["mean_correct"] for row in baseline_rows]
        baseline_any = [bool(row["any_correct"]) for row in baseline_rows]

        print(f"\nSubset Comparisons Vs Baseline: {subset_label}")
        for label, run in runs.items():
            if label == baseline_label:
                continue
            run_rows = filter_rows(run, subset_label)
            if [row["key"] for row in baseline_rows] != [row["key"] for row in run_rows]:
                raise ValueError(
                    "Subset problem alignment mismatch:\n"
                    f"  baseline={baseline['path']}\n"
                    f"  run={run['path']}\n"
                    f"  subset={subset_label}"
                )
            run_means = [row["mean_correct"] for row in run_rows]
            run_any = [bool(row["any_correct"]) for row in run_rows]
            boot = paired_bootstrap(
                baseline_means, run_means, n_boot=n_boot, seed=seed
            )
            mcnemar = exact_mcnemar(baseline_any, run_any)
            pass_delta = (sum(run_any) / len(run_any)) - (sum(baseline_any) / len(baseline_any))
            print(f"\n{baseline_label} -> {label}")
            print(
                "  Delta mean correctness (right-left): "
                f"{_round_pct(boot['delta']):+.2f}pp"
            )
            print(
                "  95% paired-bootstrap CI: "
                f"[{_round_pct(boot['ci_low']):+.2f}, {_round_pct(boot['ci_high']):+.2f}]pp"
            )
            print(
                "  Paired-bootstrap two-sided p: "
                f"{boot['p_two_sided']:.4f}"
            )
            print(f"  Delta pass@8 (right-left): {_round_pct(pass_delta):+.2f}pp")
            print(
                "  Exact McNemar p: "
                f"{mcnemar['p_value']:.4f}"
                f"  [left_only={mcnemar['left_only']}, right_only={mcnemar['right_only']}]"
            )


def main() -> None:
    args = parse_args()
    runs = discover_runs(args)
    if not runs:
        raise SystemExit(
            f"No math500_pass8_results.json files found under {args.root}"
        )

    print_run_table(runs)
    print_level_stratified_tables(runs)
    print_comparisons(
        runs,
        comparisons=[tuple(x) for x in args.compare],
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    print_subset_vs_baseline(
        runs,
        baseline_label=args.baseline_label,
        subset_labels=list(args.subset_vs_baseline),
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
