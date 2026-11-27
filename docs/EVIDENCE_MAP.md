# Claim-to-evidence map

The first path in each row is relative to the repository root; shorter paths
in that row share its experiment directory. Integrity hashes and export
origins are machine-readable in `checksums.json` and `provenance/origins.json`.
The [machine-readable claim map](../provenance/claims.json) also links individual
claims to their hashed source records.

| Paper result | Primary record | Reproduction |
|---|---|---|
| Three rules × two normalizations | `experiments/offline/results/summary.json`; `results/tables/full_geometry.csv` | `scripts/reproduce.py`, offline stage |
| Length calibration and censoring | `experiments/offline/results/calibration.json`; `censoring_report.json` | Offline stage |
| Structural opportunity and rank association | `experiments/offline/results/summary.json`; `group_metrics.csv`; `length_association.csv` | Offline stage |
| Direct learned–length residual comparison | `experiments/assignment/results/summary.json`; `per_group.csv` | Assignment stage |
| Geometry invariance for 32 reassignments | `experiments/assignment/results/permutation_checks.csv` | Assignment stage |
| Complete replay gradient comparisons | `experiments/replay/run/result.json`; `results/all_comparisons.csv` | Replay stage |
| Exact expectation, both rules | `experiments/replay/run/group_statistics.json`; `results/group_reference.csv` | Replay stage |
| All 32 learned-shuffle draws | `experiments/replay/results/permutation_draws.csv` | Replay stage |
| Token alignment and 28 one-token prompt trims | `experiments/replay/analysis/per_completion_audit.csv` | Detailed replay stage |
| Group contributions and concentration | `experiments/replay/analysis/per_group_audit.csv`; `concentration.json` | Detailed replay stage |
| Pilot versus complete sample | `experiments/replay/analysis/pilot_vs_full.csv` | Detailed replay stage |
| Alignment floors and formal identities | `experiments/validation/` | Validation stages |
| Correctness discrimination and training coverage | `experiments/historical/results/` | Historical stage |
| Original single-seed task comparison | `experiments/historical/inputs/`; `results/paired_results.json` | Saved-record comparison; original statistics code included |
| Learned-head objective and defect | `docs/method/` | Source documentation, not a new training execution |

The old 32-group pilot and earlier imported gradient diagnostic are historical
records. Use `experiments/replay/` for the manuscript's main gradient findings.
No field from a historical pilot should be substituted for the complete result.
