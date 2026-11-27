# Release verification

Verified on 27 November 2026. No model training, new decoding or paid GPU work
was performed while preparing this repository.

| Check | Outcome |
|---|---|
| CPU reproduction | PASS: 33 output files reproduced byte-for-byte |
| Standalone copied repository | PASS: the same complete workflow passed from a separate directory with spaces |
| Initial review-export scan | No researcher identifiers, private paths, credentials or broken guide links found in that initial export |
| Offline token-record validation | 4,000 saved completion records checked |
| Complete replay | 185 groups and 1,480 completion records; all 210 reference comparisons checked |
| Learned reassignments | All 32 draws retained and checked |
| Exact permutation reference | Recomputed from each group's saved gradient scatter |
| Historical statistics | 146 numerical fields checked, including task means and fixed-seed paired comparisons |
| Tiny Qwen/LoRA software test | PASS: 44 checks on CPU |
| Permutation/scatter software tests | PASS: 37 checks on CPU |
| Explicit GPU-selection guard | PASS: three CPU unit tests |
| Coefficient-cast precision | PASS: every numerical diagnostic matches the recorded result |
| Synthetic full-report workflow | PASS: all 185 synthetic groups and 32 permutation summaries |
| Fresh cluster check | Primary result, group scatter and parameter layout match the repository byte-for-byte |
| Source completeness | All 341 archive members accounted for; all 41 manuscript claim entries retained |
| Figure regeneration | All eight regenerated PDF figures match the supplied rendered pixels |

The full CPU command took about **82 seconds** on the verification machine,
excluding dependency installation. Runtime varies with CPU and disk speed.
It used Python 3.12.7, NumPy 1.26.4 and SciPy 1.15.2. Optional tiny-model tests
used the replay's pinned core package versions in a separate environment.

Machine-readable records are in `../provenance/verification/`. The complete
reproduction reports every compared output separately. The checksum verifier
can detect changes to the distributed files before recomputation.

The CPU verification does not reconstruct omitted original training, recover
missing generation-time prompt IDs, rerun 7B backpropagation, or rank learning
quality. The optional model code was tested with a tiny randomly initialized
model; the scientific 7B results remain the supplied completed replay.

The repository contains no model checkpoint, recovery bank, credential, private
server launch script, or unrelated project. Model-access limitations and the
redacted adapter locator are explicit in [MODEL_REPLAY.md](MODEL_REPLAY.md).

The expanded completeness review and rerun are documented in [COMPLETENESS.md](COMPLETENESS.md) and `../provenance/verification/completeness_cpu.json`.

## Public documentation refresh, 28 November 2026

The updated README and guides were checked in desktop and mobile previews. All 58 relative links in the new entry pages resolve locally; the public Paper and Blog links returned HTTP 200. The new normalization preview is rendered from the existing paper figure.

The complete CPU workflow passed again in 103 seconds, with all **33 numerical outputs reproduced byte-for-byte**. The record is [documentation_refresh_cpu.json](../provenance/verification/documentation_refresh_cpu.json). Experiment inputs, saved results, analysis scripts and model-replay code are unchanged. No GPU job or manuscript edit was made.

The public-facing documentation now identifies the author and links the public blog. Earlier review-export checks above retain their original scope; they are not a claim that the current public README is anonymous.
