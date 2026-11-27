# Learned-distance method

The text snapshots document the historical auxiliary objective, reward route,
negative sampling, and mixed-group filtering used in Appendix D.

- `head_objective.py.txt`: compact extracted objective.
- `lavida_trainer.py.txt`: recovered trainer at the archived source revision.
- `mrn_head.py.txt`: stabilized unsquared Mahalanobis distance.
- `hard_neg_miner.py.txt` and `multi_pos_filter.py.txt`: sampling and eligibility.
- `archived_run_manifest.json`: historical configuration and source identity.

Recovering these files does not establish the exact executed optimizer state.
The head gradients were outside normal clearing and clipping; the offline
geometry is not a causal explanation of the original task outcome. These are
reference snapshots, not a corrected training recipe.

Pinned replay weight revisions and hashes are under
`../../experiments/replay/code/data/`. See `../MODEL_REPLAY.md` for the separate
model-execution boundary. Machine paths and researcher account identifiers have
been redacted in this public export; all export changes are recorded in
`../../provenance/origins.json`.
