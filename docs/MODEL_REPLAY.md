# Executing the model replay again

The main CPU workflow verifies saved-gradient calculations. A new model replay
is a separate, optional experiment: it loads the pinned 7B base model and LoRA
adapter and differentiates the saved completions again. It does not train a
policy or redraw completions.

## Required files and access

`experiments/replay/code/data/model_pins.json` and `download_manifest.json`
record the base/adapter revisions and individual file hashes. The base is
`Qwen/Qwen2.5-Math-7B` at the recorded immutable revision. The adapter account
locator is withheld in this review export; `REPLAY_ADAPTER` is an alias, not a
Hugging Face repository. Obtain authorized access to the exact hashed adapter
files before attempting model execution. This repository does not imply that
private weights are publicly downloadable.

Stage the listed files in this layout, outside version control:

```text
model-store/
  base/       # pinned model shards, index, config and tokenizer files
  adapter/    # pinned adapter, tokenizer, run manifest and step-2000 head state
```

The preflight hashes the staged files, checks adapter/base compatibility and
verifies the frozen prompt/completion inputs and weight plans. It must pass
before a run. Do not substitute a different adapter and call it a reproduction.

## Runtime and device

The recorded package inventory is `experiments/replay/recorded_packages.txt`.
Core versions enforced by the runner include PyTorch 2.9.1, Transformers 4.57.6,
PEFT 0.19.1, NumPy 1.26.4 and Safetensors 0.7.0. The recorded CUDA-enabled
PyTorch build is 2.9.1+cu126. Use a separate environment; do not install the GPU
stack into the CPU reproduction environment unnecessarily.

The code requires at least 40 GiB of GPU memory and substantial CPU memory and
disk space for the 38-direction float32 accumulation bank. Its explicit disk
headroom checks protect resumable checkpoints. These are runtime requirements,
not an assurance that any machine meeting the minimum will reproduce timing or
floating-point accumulation exactly.

Select an authorized, idle GPU explicitly. For a Linux shell:

```bash
export GRPO_REPLAY_GPU_UUID='GPU-<the-authorized-device-uuid>'
export CUDA_VISIBLE_DEVICES=$GRPO_REPLAY_GPU_UUID
```

This is an example requiring a real device selection, not a command to launch
on another person's GPU. The runner checks both the selected visible device and
its actual UUID. The public adaptation does not hard-code the original server.

From the repository root, record the installed environment, then preflight:

```bash
python scripts/capture_runtime.py --output runtime-local.json
python experiments/replay/code/replay.py preflight --models model-store --output preflight-local.json
python experiments/replay/code/replay.py run --models model-store --native-runtime runtime-local.json --output gpu-output-canary --canary
```

The canary covers the two fixed canary groups. Inspect its finite gradients,
completed outputs and runtime identity before proceeding:

```bash
python experiments/replay/code/replay.py run --models model-store --native-runtime runtime-local.json --output gpu-output-full
python experiments/replay/analyze_result.py --run gpu-output-full --data experiments/replay/code/data --output replay-report-new
```

Use `--resume` only with the same runtime, input, source and parameter identity.
Keep the new execution record separate from the supplied primary result. A new
device/runtime execution is not a recovery of missing original prompt IDs or
independent statistical replication.

## Estimator and controls

- All 185 mixed groups; eight saved completions per group.
- Reconstructed prompt tokens; all completion tokens retained. The recorded
  28 one-token prompt trims are fixed in the input records.
- Sequence-mean completion log-probability, replay logit temperature 1.0,
  dropout disabled, LoRA parameters only; common divisor 1,480.
- Binary, learned, constant and length weightings; 32 fixed within-incorrect
  permutations of the learned penalties.
- Per-group gradient scatter for both exact conditional permutation references.
- No sampled length permutations, policy update, optimizer replay, KL gradient
  or independent task-performance evaluation.

Software tests for the optional execution route:

```bash
python experiments/replay/code/test_device_guard.py
python experiments/replay/code/test_core.py --output tiny-model-check.json
python experiments/replay/code/test_reference.py --output reference-check.json
```

The tiny randomly initialized model tests software behavior only. They are not
an additional 7B scientific result. No GPU is needed for these tests.
