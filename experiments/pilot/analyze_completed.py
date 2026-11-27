"""Descriptive tables from the complete frozen replay; no significance tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def pair(gram, a, b):
    aa, bb, ab = gram[a, a], gram[b, b], gram[a, b]
    na, nb = np.sqrt(max(0., aa)), np.sqrt(max(0., bb))
    if not na or not nb:
        return dict(cosine=None, norm_ratio=None, relative_difference=None,
                    norm_reference=float(na), norm_compared=float(nb))
    return dict(cosine=float(np.clip(ab / (na * nb), -1., 1.)),
                norm_ratio=float(nb / na),
                relative_difference=float(np.sqrt(max(0., aa + bb - 2 * ab)) / na),
                norm_reference=float(na), norm_compared=float(nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.run / 'result.json').read_text())
    assert d['status'] == 'COMPLETE' and d['groups_complete'] == 32
    assert json.loads((a.run / 'local_validation.json').read_text())['status'] == 'PASS'
    assert not a.output.exists(), 'Preserve previous analysis'
    a.output.mkdir(parents=True)
    gram = np.array(d['direction_gram'])
    basis = np.array(d['basis_gram'])
    names = d['direction_names']
    rows = []
    for j, name in enumerate(names):
        rows.append(dict(comparison=name + ' vs binary', **pair(gram, 0, j)))
    for j in range(32):
        rows.append(dict(comparison=f'shuffle_{j:02d} vs learned', **pair(gram, 1, 4+j)))
        rows.append(dict(comparison=f'residual_shuffle_{j:02d} vs learned residual',
                         **pair(gram, 36, 38+j)))
    rows.extend([
        dict(comparison='length vs learned', **pair(gram, 1, 3)),
        dict(comparison='length residual vs learned residual', **pair(gram, 36, 37)),
        dict(comparison='aligned learned vs binary', **pair(basis, 0, 1)),
        dict(comparison='aligned length vs binary', **pair(basis, 0, 2)),
    ])
    with (a.output / 'all_comparisons.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = {
        'result_sha256': hashlib.sha256((a.run / 'result.json').read_bytes()).hexdigest(),
        'primary': {names[j]: pair(gram, 0, j) for j in [1, 2, 3]},
        'length_vs_learned': pair(gram, 1, 3),
        'length_residual_vs_learned_residual': pair(gram, 36, 37),
        'aligned_learned_vs_binary': pair(basis, 0, 1),
        'aligned_length_vs_binary': pair(basis, 0, 2),
        'shuffle_summary': {},
        'groups': 32, 'completions': 256, 'permutation_draws': 32,
        'scope': d['identity']['estimator'],
        'boundary': 'Fixed-sample assignment sensitivity, not learning quality or a historical-update reconstruction.',
    }
    for label, ref, ix in [('full_gradient_vs_binary', 0, range(4,36)),
                           ('full_gradient_vs_learned', 1, range(4,36)),
                           ('residual_gradient_vs_original', 36, range(38,70))]:
        values = [pair(gram, ref, j) for j in ix]
        report['shuffle_summary'][label] = {}
        for key in ['cosine', 'norm_ratio', 'relative_difference']:
            valid = [v[key] for v in values if v[key] is not None]
            quantiles = np.quantile(valid, [0,.25,.5,.75,1]).tolist() if valid else [None]*5
            report['shuffle_summary'][label][key] = dict(
                defined=len(valid), excluded=32-len(valid),
                **dict(zip(['min','q1','median','q3','max'], quantiles)))
    (a.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    remote = json.loads((a.run / 'summary.json').read_text())
    for label, family in report['shuffle_summary'].items():
        for key, values in family.items():
            for k, value in values.items():
                other = remote['shuffle_summary'][label][key][k]
                if value is None:
                    assert other is None
                else:
                    assert np.isclose(value, other, atol=1e-10, rtol=1e-9)
    (a.output / 'summary_validation.json').write_text(json.dumps({
        'status': 'PASS', 'comparison_rows': len(rows),
        'all_32_draws_reported': True,
        'scope': 'Independent arithmetic from the retrieved Gram matrix, not an independent model execution.'
    }, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
