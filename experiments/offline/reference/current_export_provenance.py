"""Deterministic hashing and runtime provenance helpers for evaluation.

The helpers in this module deliberately avoid importing vLLM.  This keeps the
planning and smoke-test path CPU-only while allowing ``evaluate_math.py`` to
record enough information to identify an evaluation stack exactly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SHA1_HEX_LENGTH = 40
SHA256_HEX_LENGTH = 64
CODE_PREFIXES = ("lavida/", "scripts/", "configs/", "training/")
CODE_ROOT_SUFFIXES = (".py", ".toml", ".yaml", ".yml", ".json", ".txt")
CRITICAL_SOURCE_SUFFIXES = frozenset(CODE_ROOT_SUFFIXES)


def utc_now_iso() -> str:
    """Return an RFC-3339 UTC timestamp without platform-specific formatting."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON using a canonical, UTF-8 serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def sha256_tree(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash a file or a directory tree, including relative names and sizes.

    The aggregate digest commits to each regular file's relative POSIX path,
    byte length, and SHA-256. Symlinks commit to both their link target and the
    resolved target content, preventing cache-backed artifacts from changing
    behind an unchanged link.
    """

    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise FileNotFoundError(f"Cannot hash missing artifact: {root}")
    if root.is_file():
        file_hash = sha256_file(root)
        return {
            "kind": "file",
            "sha256": file_hash,
            "file_count": 1,
            "total_bytes": root.stat().st_size,
            "entries": [
                {
                    "path": root.name,
                    "size": root.stat().st_size,
                    "sha256": file_hash,
                }
            ],
        }

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            target = os.readlink(child)
            resolved = child.resolve(strict=True)
            if resolved.is_file():
                resolved_artifact = {
                    "kind": "file",
                    "size": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            elif resolved.is_dir():
                nested = sha256_tree(resolved)
                resolved_artifact = {
                    "kind": "directory",
                    "file_count": nested["file_count"],
                    "total_bytes": nested["total_bytes"],
                    "sha256": nested["sha256"],
                }
            else:
                raise ValueError(f"Unsupported symlink target: {child} -> {resolved}")
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "resolved_artifact": resolved_artifact,
                    "sha256": sha256_json(
                        {"target": target, "resolved_artifact": resolved_artifact}
                    ),
                }
            )
            continue
        if not child.is_file():
            continue
        size = child.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "size": size,
                "sha256": sha256_file(child),
            }
        )

    return {
        "kind": "directory",
        "sha256": sha256_json(entries),
        "file_count": sum(entry.get("kind") == "file" for entry in entries),
        "total_bytes": total_bytes,
        "entries": entries,
    }


def sha256_source_tree(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash a source directory while excluding generated/cache metadata."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot hash missing source directory: {root}")
    excluded_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    excluded_suffixes = {".pyc", ".pyo"}
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative_path = child.relative_to(root)
        if any(part in excluded_dirs for part in relative_path.parts):
            continue
        if child.suffix.lower() in excluded_suffixes or child.name == ".DS_Store":
            continue
        if child.is_symlink():
            target = os.readlink(child)
            resolved = child.resolve(strict=True)
            resolved_sha256 = (
                sha256_file(resolved)
                if resolved.is_file()
                else sha256_tree(resolved)["sha256"]
            )
            entries.append(
                {
                    "path": relative_path.as_posix(),
                    "kind": "symlink",
                    "target": target,
                    "resolved_sha256": resolved_sha256,
                    "sha256": sha256_json(
                        {"target": target, "resolved_sha256": resolved_sha256}
                    ),
                }
            )
        elif child.is_file():
            size = child.stat().st_size
            total_bytes += size
            entries.append(
                {
                    "path": relative_path.as_posix(),
                    "kind": "file",
                    "size": size,
                    "sha256": sha256_file(child),
                }
            )
    return {
        "kind": "filtered_source_directory",
        "sha256": sha256_json(entries),
        "file_count": sum(entry.get("kind") == "file" for entry in entries),
        "total_bytes": total_bytes,
        "excluded_directories": sorted(excluded_dirs),
        "excluded_suffixes": sorted(excluded_suffixes),
        "entries": entries,
    }


def critical_source_files(
    repo_root: str | os.PathLike[str],
    source_roots: Iterable[str],
) -> list[str]:
    """Enumerate every lock-eligible file below declared source roots.

    This deliberately mirrors the protocol locker's suffix and cache rules.
    It detects newly added executable/config files that are absent from an
    older critical-file map.
    """

    root = Path(repo_root).resolve()
    relatives: set[str] = set()
    for source_root_text in source_roots:
        source_root = (root / str(source_root_text)).resolve()
        try:
            source_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Critical source root escapes repository: {source_root_text}"
            ) from exc
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"Critical source root is missing: {source_root_text}"
            )
        for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
            if (
                not path.is_file()
                or path.suffix.lower() not in CRITICAL_SOURCE_SUFFIXES
                or "__pycache__" in path.parts
            ):
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Critical source file escapes repository: {path}"
                ) from exc
            relatives.add(path.relative_to(root).as_posix())
    return sorted(relatives)


def verify_critical_source_closure(
    repo_root: str | os.PathLike[str],
    source_roots: Iterable[str],
    locked_relative_paths: Iterable[str],
) -> dict[str, Any]:
    """Reject source-root files that were not committed by the protocol lock."""

    roots = [str(root) for root in source_roots]
    if not roots:
        raise ValueError("Protocol lock has no critical source roots")
    eligible = critical_source_files(repo_root, roots)
    locked = {str(path).replace("\\", "/") for path in locked_relative_paths}
    unlocked = sorted(set(eligible).difference(locked))
    if unlocked:
        raise ValueError(
            "Critical source closure contains files absent from the lock: "
            + ", ".join(unlocked)
        )
    return {
        "source_roots": roots,
        "eligible_file_count": len(eligible),
        "eligible_relative_paths_sha256": sha256_json(eligible),
        "unlocked_files": [],
    }


def _run_git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _is_code_path(status_path: str) -> bool:
    # Rename entries are formatted as ``old -> new``.  The new path is what
    # could be imported or executed, so use it for the relevance check.
    normalized = status_path.replace("\\", "/").split(" -> ")[-1]
    return normalized.startswith(CODE_PREFIXES) or (
        "/" not in normalized and normalized.endswith(CODE_ROOT_SUFFIXES)
    )


def git_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return commit and dirty-state metadata for the enclosing Git worktree."""

    requested = Path(path).resolve()
    top = _run_git(requested, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return {
            "available": False,
            "error": (top.stderr or top.stdout).strip(),
        }
    root = Path(top.stdout.strip()).resolve()
    commit_proc = _run_git(root, "rev-parse", "HEAD")
    status_proc = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    diff_proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    status_lines = [line for line in status_proc.stdout.splitlines() if line]
    relevant_lines = [
        line
        for line in status_lines
        if _is_code_path(line[3:] if len(line) > 3 else "")
    ]
    return {
        "available": commit_proc.returncode == 0 and status_proc.returncode == 0,
        "worktree_root": str(root),
        "commit": commit_proc.stdout.strip() or None,
        "dirty": bool(status_lines),
        "code_dirty": bool(relevant_lines),
        "status": status_lines,
        "code_status": relevant_lines,
        "tracked_diff_sha256": (
            sha256_bytes(diff_proc.stdout) if diff_proc.returncode == 0 else None
        ),
    }


def package_versions(distributions: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _nvidia_driver_versions() -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def runtime_snapshot(
    container_image_digest: str | None,
    modal_image_id: str | None = None,
) -> dict[str, Any]:
    """Capture versions and accelerator facts without mutating the environment."""

    packages = package_versions(
        (
            "torch",
            "vllm",
            "transformers",
            "tokenizers",
            "datasets",
            "huggingface-hub",
            "peft",
            "trl",
            "accelerate",
            "numpy",
            "scipy",
            "Pebble",
        )
    )
    torch_info: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": device_count,
            "devices": [
                torch.cuda.get_device_name(index) for index in range(device_count)
            ],
        }
    except Exception as exc:  # pragma: no cover - depends on the runtime image
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "captured_at_utc": utc_now_iso(),
        "container_image_digest": container_image_digest,
        "modal_image_id": modal_image_id or os.environ.get("MODAL_IMAGE_ID"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch": torch_info,
        "nvidia_driver_versions": _nvidia_driver_versions(),
    }


def canonical_item_id(item: dict[str, Any]) -> tuple[str, str]:
    """Return a stable evaluation ID and describe how it was constructed."""

    if item.get("unique_id") is not None:
        return str(item["unique_id"]), "dataset_unique_id"
    canonical = {
        "problem": str(item.get("problem", "")),
        "answer": str(item.get("answer", "")),
    }
    return f"sha256:{sha256_json(canonical)}", "canonical_problem_answer_sha256"


def ordered_item_provenance(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids_and_sources = [canonical_item_id(item) for item in items]
    ordered_ids = [item_id for item_id, _ in ids_and_sources]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Evaluation items do not have unique canonical IDs")
    sources = sorted({source for _, source in ids_and_sources})
    return {
        "count": len(ordered_ids),
        "id_sources": sources,
        "ordered_item_ids_sha256": sha256_json(ordered_ids),
        "ordered_item_ids": ordered_ids,
    }


def prompt_provenance(prompts: list[str]) -> dict[str, Any]:
    prompt_hashes = [sha256_bytes(prompt.encode("utf-8")) for prompt in prompts]
    return {
        "count": len(prompt_hashes),
        "ordered_prompt_sha256": sha256_json(prompt_hashes),
        "prompt_sha256": prompt_hashes,
    }


def completion_output_record(output: Any) -> dict[str, Any]:
    """Serialize the reproducibility fields of a vLLM CompletionOutput.

    ``cumulative_logprob`` is divided by the number of emitted token IDs.  The
    resulting mean is therefore a per-generated-token log probability.  Empty
    generations and runtimes that do not expose cumulative log probability are
    represented by ``None`` rather than a fabricated score.
    """

    token_ids = [
        int(token_id) for token_id in (getattr(output, "token_ids", None) or [])
    ]
    cumulative = getattr(output, "cumulative_logprob", None)
    cumulative_value = float(cumulative) if cumulative is not None else None
    if cumulative_value is not None and not math.isfinite(cumulative_value):
        cumulative_value = None
    mean_logprob = (
        cumulative_value / len(token_ids)
        if cumulative_value is not None and token_ids
        else None
    )
    stop_reason = getattr(output, "stop_reason", None)
    if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
        stop_reason = str(stop_reason)
    finish_reason = getattr(output, "finish_reason", None)
    if finish_reason is not None and not isinstance(
        finish_reason, (str, int, float, bool)
    ):
        finish_reason = str(finish_reason)
    return {
        "completion": str(getattr(output, "text", "")),
        "token_ids": token_ids,
        "num_generated_tokens": len(token_ids),
        "cumulative_logprob": cumulative_value,
        "mean_logprob": mean_logprob,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
    }


def is_full_git_revision(value: str | None) -> bool:
    if value is None or len(value) != SHA1_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def is_immutable_revision(value: str | None) -> bool:
    """Accept full immutable Hub revisions (40-64 hexadecimal characters)."""

    if value is None or not (SHA1_HEX_LENGTH <= len(value) <= SHA256_HEX_LENGTH):
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def is_sha256(value: str | None) -> bool:
    if value is None or len(value) != SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def normalize_container_digest(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.rsplit("@", 1)[-1]
    if digest.startswith("sha256:") and is_sha256(digest.removeprefix("sha256:")):
        return value
    return None
