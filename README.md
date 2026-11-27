# What Survives Group Normalization?

**A Credit-Assignment Audit of Reward Shaping in GRPO**

A shaping score can separate correct and incorrect answers without adding much
new direction after group normalization. Even when it changes the advantages,
their geometry does not tell us whether the right completions received credit.
This repository contains the code and saved evidence behind that distinction.

![Residual-gradient scale and the exact permutation reference](figures/preview.png)

The figure covers all 185 mixed-outcome groups in one saved MATH-500 sample.
Gray points reassign the learned penalties among incorrect completions while
preserving their advantage geometry. These are gradient measurements, not
measurements of improved learning.

## Start here

Python **3.12** is the tested version. The default workflow is CPU-only and
uses bundled files. It needs no model weights, credentials, GPU, or network
connection after dependencies are installed.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_repository.py
python scripts/reproduce.py
```

Results and logs go into `reproduced/`; committed evidence is never overwritten.
Choose a new destination for a repeat, for example
`python scripts/reproduce.py --output reproduced-again`.
The [reproduction guide](docs/REPRODUCING.md) explains what each stage checks.

## What the evidence shows

| Measurement | Result | Population |
|---|---:|---|
| Learned–binary advantage cosine | 0.9786 | 500 prompts, 8 completions each |
| Learned–length advantage cosine | 0.9684 | Same saved sample |
| Learned–length within-group residual cosine | 0.2303 | Same saved sample |
| Prompts permitting within-incorrect differentiation | 132 / 500 | 185 mixed groups, including 53 with one incorrect completion |
| Learned–binary model-gradient cosine | 0.9844 | All 185 mixed groups, 1,480 completions |
| Learned residual norm / binary-gradient norm | 17.48% | Same complete replay |
| Learned–length residual-gradient cosine | −0.0026 | Same complete replay |
| Learned squared residual norm / exact permutation expectation | 1.0824× | Conditional on fixed computed gradients |
| Length squared residual norm / its own exact expectation | 2.3213× | Conditional on the same computed gradients |

The learned ratio lies within the range of its 32 sampled reassignments
(0.8994–1.1500); four draws exceed the observed norm. No length-penalty shuffles
were collected. Its 2.3213× ratio is descriptive, with no significance or
learning-quality claim. Length also enters sequence-mean gradient weighting;
the replay does not isolate that interaction.

## Repository map

| Directory | Contents |
|---|---|
| [`experiments/offline/`](experiments/offline/) | Three reward rules under two normalization conventions, saved tokens, sensitivities and results |
| [`experiments/assignment/`](experiments/assignment/) | Direct residual comparisons, conditional distributions and 32 permutation checks |
| [`experiments/replay/`](experiments/replay/) | Complete 185-group model-gradient results, exact reference, execution code and pinned inputs |
| [`experiments/pilot/`](experiments/pilot/) | Earlier 32-group replay, retained only as a labelled subset comparison |
| [`experiments/validation/`](experiments/validation/) | Synthetic and saved-data checks of the decomposition, alignment floor and permutation identity |
| [`experiments/historical/`](experiments/historical/) | Motivating task evaluation, AUROC and training-coverage records |
| [`docs/method/`](docs/method/) | Recovered historical objective and trainer snapshots |
| [`figures/`](figures/) | Paper figures, editable SVGs and a portable generator |
| [`provenance/`](provenance/) | Original seal records and the public-export change map |

For individual claims, use the [evidence map](docs/EVIDENCE_MAP.md). For a new
model execution, read [Model replay](docs/MODEL_REPLAY.md) first. The CPU command
above recomputes saved-gradient arithmetic; it does not rerun backpropagation.

## Scope

This is a fixed-sample audit of Qwen2.5-Math-7B with LoRA. It includes the
completed model replay, but not corrected multi-seed training or new decoding.
Historical prompt token IDs were not saved, so replay prompts were reconstructed.
The original learned head had a gradient-clearing/clipping defect. Neither the
original task comparison nor this offline analysis isolates normalization as
the cause of the training outcome.

The original evaluation, fresh generation sample, AUROC cohort and training
coverage have different denominators. They remain separate throughout the
[scope notes](docs/SCOPE.md).

Public-export seals check file integrity after packaging. They are **not new
preregistrations**. Original numerical records are retained, while personal
paths and account identifiers are redacted and duplicate transport files are
excluded. See [provenance](docs/PROVENANCE.md) and the
[verification report](docs/VERIFICATION.md). The [completeness review](docs/COMPLETENESS.md)
accounts for the local archive, fresh cluster checks, and deliberate exclusions.

This is the code-and-evidence companion, not the manuscript source project.
Third-party attribution is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code and code examples are available under the [MIT License](LICENSE).
Original explanatory text and figures are available under
[CC BY 4.0](LICENSE-CONTENT.md), with attribution required when shared.
Third-party code, dataset material and model-related files retain their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
