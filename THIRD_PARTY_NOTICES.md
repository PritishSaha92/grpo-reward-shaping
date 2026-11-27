# Third-party material and attribution

The repository's original code uses the MIT License in `LICENSE`. Original
explanatory text and figures use CC BY 4.0 as specified in `LICENSE-CONTENT.md`.
These grants apply only to material the project contributors have the right to
license; they do not replace the third-party terms described below.

The prompt construction helpers in `experiments/replay/code/data/prompt_source/`
come from the Qwen2.5-Math evaluation code. The upstream evaluation license is
included verbatim in `licenses/QWEN_EVALUATION_MIT.txt`; its copyright notice
must be retained. The preserved helper hashes identify the exact source used.

MATH problem material and model-generated completions appear in the saved
research records. Their inclusion here does not relicense the underlying
dataset. Relevant sources are:

- Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH
  Dataset*, NeurIPS Datasets and Benchmarks, 2021.
- Lightman et al., *Let's Verify Step by Step*, 2023, for the MATH-500 evaluation
  split used by the study.
- Qwen2.5-Math, for the base policy and evaluation utilities.

Model weights are not included. Base-model, tokenizer, adapter and dataset
terms continue to apply separately. Python dependencies are installed from
their distributions and retain their own licenses.

Historical trainer snapshots document the method and its defect. They contain
interfaces to TRL, PyTorch, PEFT and vLLM; they are not vendored copies of those
libraries or a new claim of authorship over their implementations.

This export retains third-party attribution even when researcher account names
and machine paths are removed. Anonymization should not remove license notices.

The unmodified CC BY 4.0 legal text in `licenses/CC-BY-4.0.txt` was retrieved
from the SPDX license-list-data repository. Creative Commons dedicates its
license text to the public domain under CC0; see its
[licensing statement](https://creativecommons.org/policies/).
The [canonical CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
governs the original content identified in `LICENSE-CONTENT.md`.
