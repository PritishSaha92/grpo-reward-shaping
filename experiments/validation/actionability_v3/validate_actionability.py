"""Post-result identity checks and descriptive coefficient-independent floors."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=ROOT.parents[1] / 'offline')
parser.add_argument('--output', type=Path, default=ROOT)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261126)
counts, errors, failures = defaultdict(int), [], []
undefined = 0

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(kind, actual, expected):
    counts[kind] += 1
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    errors.append(error)
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-8):
        failures.append({'check': kind, 'absolute_error': error})

def center(z):
    return z-z.mean()

def advantages(z, mode):
    c = center(z)
    if mode == 'mean':
        return c
    denom = z.std() + float(mode)
    return np.zeros_like(z) if denom == 0 else c/denom

def cosine(a, b):
    return float(a@b / np.linalg.norm(a) / np.linalg.norm(b))

def permute(R, g):
    idx = np.arange(len(R))
    for outcome in (0, 1):
        take = np.flatnonzero(R == outcome)
        idx[take] = rng.permutation(take)
    return g[idx]

def check_group(R, g, mode):
    global undefined
    A, B = advantages(R, mode), advantages(R+g, mode)
    if np.linalg.norm(B) < 1e-12:
        undefined += 1
        return
    x, y = center(R), center(R+g)
    c = cosine(A, B)
    check('pearson', c, np.corrcoef(R, R+g)[0, 1])
    w = g.copy()
    for outcome in (0, 1):
        w[R == outcome] -= g[R == outcome].mean()
    check('unexplained_variance', 1-c*c, w@w/(y@y))
    gp = permute(R, g)
    Bp = advantages(R+gp, mode)
    check('per_group_permutation_cosine', cosine(A, Bp), c)
    alpha = A@B/(A@A)
    W = B-alpha*A
    H = rng.normal(size=(len(R), 5))
    Hc = H.copy()
    for outcome in (0, 1):
        take = R == outcome
        check('class_residual_sum', W[take].sum(), 0.)
        Hc[take] -= H[take].mean(axis=0)
    check('residual_gradient_contrast', W@H, W@Hc)
    check('full_gradient_difference', (B-A)@H, (alpha-1)*(A@H)+W@Hc)

def pooled(groups, mode):
    aa, bb, ww, aligned = [], [], [], []
    for R, g in groups:
        A, B = advantages(R, mode), advantages(R+g, mode)
        if A@A == 0:
            assert B@B == 0  # applies to the gated saved-ledger rules
            continue
        alpha = A@B/(A@A)
        aa.append(A); bb.append(B); ww.append(B-alpha*A)
        aligned.append((alpha, A@A))
    A, B = np.concatenate(aa), np.concatenate(bb)
    c = cosine(A, B)
    alpha = A@B/(A@A)
    within = sum(w@w for w in ww)/(B@B)
    weight = sum((a-alpha)**2*n for a, n in aligned)/(B@B)
    return np.array([c, np.linalg.norm(B)/np.linalg.norm(A),
                     np.linalg.norm(B-A)/np.linalg.norm(A), 1-c*c,
                     within, weight])

for k in range(2, 33):
    for n_correct in range(1, k):
        R = np.r_[np.ones(n_correct), np.zeros(k-n_correct)]
        g = rng.normal(size=k)
        residual = g.copy()
        for outcome in (0, 1):
            residual[R == outcome] -= g[R == outcome].mean()
        cases = [g, -(1-R)*rng.uniform(0, 4, k), R*rng.uniform(0, 4, k),
                 residual-3*R, residual-R, -R]
        for shaping in cases:
            for mode in ('mean', 0., 1e-4):
                check_group(R, shaping, mode)

ledger = args.run/'joined_rollouts.jsonl'
assert sha(ledger) == 'c74f8705d3df940ea7f697d36747d2525ee187152ed709a5ca3e54a6a223a4a0'
groups = defaultdict(list)
for line in ledger.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    groups[row['group_id']].append(row)
assert len(groups) == 500 and sum(map(len, groups.values())) == 4000
eligible = []
for gid, rows in sorted(groups.items()):
    rows.sort(key=lambda r: r['sample_idx'])
    assert len(rows) == 8
    R = np.array([r['correctness'] for r in rows], dtype=float)
    if 0 < R.sum() < len(R):
        eligible.append((gid, rows, R))
assert len(eligible) == 185
qsum = sum(r['learned_penalty'] for _, rows, _ in eligible for r in rows)
lsum = sum(r['stored_generated_token_count'] for _, rows, _ in eligible
           for r in rows if not r['correctness'])
lam = qsum/lsum
rules = {key: [] for key in ('learned', 'constant', 'length_global', 'length_per_group')}
floor_rows = []
for gid, rows, R in eligible:
    wrong = R == 0
    q = np.array([r['learned_penalty'] for r in rows])
    length = np.array([r['stored_generated_token_count'] for r in rows], dtype=float)
    distance = np.array([r['deployed_distance'] for r in rows], dtype=float)
    penalties = {'learned': q, 'constant': wrong*q[wrong].mean(),
                 'length_global': wrong*lam*length,
                 'length_per_group': wrong*length*q.sum()/length[wrong].sum()}
    for rule, penalty in penalties.items():
        rules[rule].append((R, -penalty))
        for mode in ('mean', 0., 1e-4):
            check_group(R, -penalty, mode)
    for score, d in [('learned_distance', distance), ('token_length', length)]:
        values = d[wrong]
        assert np.isfinite(values).all() and (values >= 0).all()
        mu, sd, p = float(values.mean()), float(values.std()), float(R.mean())
        cv = sd/mu if mu else None
        floor = float(np.sqrt(p/(p+cv*cv))) if mu else 1.
        for beta in (0., .1, 1., 10., 1e6):
            direct = cosine(center(R), center(R-beta*wrong*d))
            tau = beta*sd/(1+beta*mu)
            expected = float(np.sqrt(p/(p+tau*tau)))
            check('floor_formula', direct, expected)
            check('floor_inequality', max(0., floor-direct), 0.)
        if mu:
            check('infinite_coefficient_limit', cosine(center(R), center(-(wrong*d))), floor)
        else:
            check('all_zero_score', sd, 0.)
        floor_rows.append({'group_id': gid, 'score': score, 'p': p,
                           'n_incorrect': int(wrong.sum()), 'mu0': mu,
                           'sigma0': sd, 'cv0': cv, 'limiting_cosine': floor})

for rule, gs in rules.items():
    permuted = [(R, permute(R, g)) for R, g in gs]
    for mode in ('mean', 0., 1e-4):
        check('pooled_permutation_six_statistics', pooled(gs, mode), pooled(permuted, mode))

# Fixed assignment example: same moments, same geometry, different gradient.
R = np.array([1., 1., 0., 0.])
g = np.array([0., 0., -1., -3.])
gp = np.array([0., 0., -3., -1.])
H = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
A, B, Bp = advantages(R, 0.), advantages(R+g, 0.), advantages(R+gp, 0.)
W, Wp = B-(A@B/(A@A))*A, Bp-(A@Bp/(A@A))*A
check('example_geometry', pooled([(R, g)], 0.), pooled([(R, gp)], 0.))
check('example_opposite_residual_gradients', W@H, -(Wp@H))
assert np.linalg.norm((W-Wp)@H) > .1

summary = {}
for score in ('learned_distance', 'token_length'):
    rr = [r for r in floor_rows if r['score'] == score]
    f = [r['limiting_cosine'] for r in rr]
    summary[score] = dict(zip(['min', 'q1', 'median', 'q3', 'max'],
                             map(float, np.quantile(f, [0, .25, .5, .75, 1]))))
    summary[score].update(groups=len(rr), zero_mean=sum(r['mu0'] == 0 for r in rr),
                          single_incorrect=sum(r['n_incorrect'] == 1 for r in rr))
with (args.output/'alignment_floors.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(floor_rows[0]))
    writer.writeheader(); writer.writerows(floor_rows)
report = {'status': 'PASS' if not failures else 'FAIL',
          'scope': 'Post-result algebra validation and new descriptive finite-sample floor summary; no training.',
          'checks': sum(counts.values()), 'checks_by_kind': dict(counts),
          'max_absolute_error': max(errors), 'failures': failures,
          'degenerate_cosines_excluded': undefined,
          'atol': 1e-10, 'rtol': 1e-8, 'random_seed': 20261126,
          'mixed_groups': 185, 'homogeneous_groups_excluded': 315,
          'floor_summary': summary,
          'permutation_example': {'original_residual_gradient': (W@H).tolist(),
                                  'permuted_residual_gradient': (Wp@H).tolist()},
          'hashes': {'ledger': sha(ledger), 'original_seal': sha(args.run/'seal.json'),
                     'amendment': sha(ROOT/'AMENDMENT.md'), 'code': sha(Path(__file__)),
                     'floor_csv': sha(args.output/'alignment_floors.csv')}}
(args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
assert not failures
