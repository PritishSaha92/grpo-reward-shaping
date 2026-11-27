# Public-export provenance

This repository is a curated export of the completed paper evidence. Duplicate
output trees, watcher/transfer logs, machine launch scripts, private paths,
researcher account identifiers, and model/recovery weight banks are excluded or
redacted. Original research folders and evidence archives are unchanged.

`provenance/origins.json` records the original archive member, its SHA-256, the
exported SHA-256 and whether the content changed. Numeric ledgers, token arrays,
weight plans, Gram matrices and group scatter retain their scientific values.
The export is checked against the original numerical reports.

Three kinds of integrity records must be distinguished:

- **Original seals:** preserved historical file-hash records. Some hashes refer
  to unredacted originals that are not redistributed. These records are not an
  exhaustive manifest of the public repository.
- **Export seals:** identify the public versions used by the portable scripts.
  Their metadata explicitly says the export happened after results were known.
- **Repository manifest:** `checksums.json` exhaustively identifies delivered
  files. It is an integrity check, not external timestamping or preregistration.

The portable code changes path resolution and integrity checks to accommodate
the redacted export. The complete replay's hardware guard now requires the
operator to explicitly supply an authorized GPU UUID, instead of embedding a
private machine's selection in runnable code. No gradient estimator, reward
coefficient, cohort, permutation map, or measured numerical result was changed.

Transport-log verification is not repeated by the public detailed-analysis
script. That script instead validates scientific records, token accounting,
matrix consistency and the conditional reference. The original extraction audit
also checked transport logs; the repository does not claim those omitted files
are reproducible from this export.

The adapter's private account locator is replaced by the explicit alias
`REPLAY_ADAPTER`; its revision and file hashes remain available. This alias is
not a downloadable model name. CPU verification does not require it. See
[MODEL_REPLAY.md](MODEL_REPLAY.md).
