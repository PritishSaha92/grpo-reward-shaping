# Repository completeness

This export supports the paper's saved-data analyses and documents the completed
model-gradient replay. A fresh completeness review on 27 November 2026 checked
the local evidence archive and retrieved the primary result records again from
the execution cluster. No new training, decoding or GPU replay was started.

## What was checked

- Every one of the **341 members** in the original paper evidence archive has an
  inclusion, duplicate, replacement or exclusion entry in
  [source_inventory.json](../provenance/source_inventory.json).
- The freshly retrieved full-replay result, per-group gradient statistics and
  parameter layout match the repository files byte-for-byte. The execution
  identity retains the same numerical fields after private metadata redaction.
- The cluster's original compact export checksum matches the archive already
  fetched locally. It contains the completed 185-group execution.
- The original adapter, step-2000 head state, tokenizers and model metadata were
  checked locally against all 16 download-manifest entries. They remain outside
  Git; public base-model shards are identified by immutable revision and hashes.
- All 41 mapped manuscript evidence entries resolve to hashed repository files.

The checks are recorded in
[cluster_completeness.json](../provenance/verification/cluster_completeness.json),
[claims.json](../provenance/claims.json), and the
[verification report](VERIFICATION.md).

## Included scientific material

| Part | Included material |
|---|---|
| Offline audit | 4,000 completion records, token records, calibration, three reward rules, both normalization conventions and sensitivities |
| Assignment comparison | Direct advantage/residual comparisons, conditional group distributions and all 32 learned-penalty reassignments |
| Complete gradient replay | 185 groups, 1,480 completions, aggregate gradient Gram matrices, per-group scatter, exact references, model/input identity and runnable replay code |
| Pilot | Saved 32-group result and its arithmetic analyzer, explicitly separated from the complete replay |
| Historical motivation | Saved task evaluations, paired statistics, AUROC inputs and training coverage |
| Method | Recovered trainer, exact head objective, metric, sampling and group-eligibility source snapshots |
| Validation | Algebraic checks, precision diagnostic, synthetic full-report test and optional tiny-model/gradient software tests |
| Presentation | Eight current paper figures, editable SVGs and a portable figure generator |

## Deliberate exclusions and limits

| Excluded material | Reason and consequence |
|---|---|
| Duplicate outputs, old figure exports and build caches | Current numerical outputs and paper figures are retained once. |
| Transport logs, private launch scripts and machine paths | They are not inputs to the numerical analyses. Source/model identity and scientific records remain available. |
| Two large gradient recovery banks | These resumable vector accumulators remain on the cluster. The included aggregate Gram matrices and group scatter suffice for the reported statistics; arbitrary new gradient measurements may require another replay. |
| Adapter/head weights and base-model shards | Keep large model files outside Git. Re-executing the 7B model requires access to the exact pinned files; see [MODEL_REPLAY.md](MODEL_REPLAY.md). The neutral adapter alias is not a download URL. |
| Manuscript source | The current named-author PDF is in `paper/`; editable LaTeX is supplied separately as the author's Overleaf bundle. |
| Missing historical prompt IDs or execution state | These were not recovered. Their absence is disclosed; repository completeness cannot recreate unavailable original records. |

The CPU reproduction is self-contained after dependency installation. It checks
saved results and does not re-create the original training trajectory. Publishing
code does not provide access to restricted model weights; access is a separate
requirement if new model execution is desired.
