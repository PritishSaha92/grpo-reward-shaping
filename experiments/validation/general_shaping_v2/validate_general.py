"""Post-result algebra checks, including a synthetic clipped-surrogate gradient."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=ROOT.parents[1] / "offline")
args = parser.parse_args()
rng = np.random.default_rng(20261126)
errors, failures, count = [], [], 0
fd_errors = []

def check(label, actual, expected, fd=False):
    global count
    count += 1
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    (fd_errors if fd else errors).append(error)
    if not np.allclose(actual, expected, atol=1e-7 if fd else 1e-10,
                       rtol=1e-6 if fd else 1e-8):
        failures.append({"label": label, "max_absolute_error": error})

def center(z):
    return z-z.mean()

def norm(z, mode):
    x = center(z)
    return x if mode == "mean" else x/(z.std()+mode)

def cosine(a, b):
    return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))

for k in range(2, 33):
    for n_correct in range(1, k):
        R = np.r_[np.ones(n_correct), np.zeros(k-n_correct)]
        p = R.mean()
        x = center(R)
        for case in ("signed", "negative", "zero", "positive", "degenerate",
                     "incorrect", "correct"):
            g = rng.normal(size=k)
            if case in ("negative", "zero", "positive", "degenerate"):
                s_target = {"negative": -2., "zero": 0.,
                            "positive": 2., "degenerate": 0.}[case]
                g = (np.zeros(k) if case == "degenerate" else
                     g-np.where(R == 1, g[R == 1].mean(), g[R == 0].mean()))
                g = g+(s_target-1)*R
            elif case == "incorrect":
                g = -(1-R)*rng.uniform(0, 4, k)
            elif case == "correct":
                g = R*rng.uniform(0, 4, k)
            mu = [g[R == r].mean() for r in (0, 1)]
            var = [g[R == r].var() for r in (0, 1)]
            s = 1+mu[1]-mu[0]
            w = g-np.where(R == 1, mu[1], mu[0])
            y = center(R+g)
            check("decomposition", y, s*x+w)
            check("orthogonality", x@w, 0.)
            V = p*var[1]+(1-p)*var[0]
            check("within_class_variance", w@w/k, V)
            if np.linalg.norm(y) < 1e-12:
                check("zero_vector_condition", s*s*p*(1-p)+V, 0.)
                continue
            expected = s*np.sqrt(p*(1-p))/np.sqrt(s*s*p*(1-p)+V)
            for mode in ("mean", 0., 1e-4):
                check("signed_angle", cosine(norm(R+g, mode), norm(R, mode)), expected)
            if case == "incorrect":
                q = -g[R == 0]
                tau = q.std()/(1+q.mean())
                check("incorrect_corollary", expected**2, p/(p+tau*tau))
            if case == "correct":
                b = g[R == 1]
                tau = b.std()/(1+b.mean())
                check("correct_corollary", expected**2, (1-p)/(1-p+tau*tau))

# A differentiable local ratio model exp(theta dot h_i); clipping inactive at zero.
R = np.array([1., 1., 0., 0.])
H = rng.normal(size=(4, 7))
A = norm(R, 0.)
B = norm(R-np.array([0., 0., 1., 3.]), 0.)
check("fixed_feature_difference", H.T@B-H.T@A, H.T@(B-A))
def surrogate(theta, advantage):
    ratio = np.exp(H@theta)
    return np.minimum(ratio*advantage, np.clip(ratio, .8, 1.2)*advantage).sum()
for advantage in (A, B):
    gradient = np.zeros(7)
    for j in range(7):
        delta = np.zeros(7); delta[j] = 1e-6
        gradient[j] = (surrogate(delta, advantage)-surrogate(-delta, advantage))/(2e-6)
    check("clipped_surrogate_at_ratio_one", gradient, H.T@advantage, fd=True)
constant = R-2*(1-R)
check("constant_ideal_gradient", H.T@norm(constant, 0.), H.T@A)
check("constant_mean_gradient", H.T@center(constant), 3*(H.T@center(R)))
check("worked_example_cosine", cosine(A, B), 3/np.sqrt(11))
ledger = args.run/"joined_rollouts.jsonl"
records = [json.loads(line) for line in ledger.read_text().splitlines()]
eligible = [r for r in records if r["group_type"] == "mixed" and r["correctness"] == 0]
positive = sum(float(r["learned_penalty"]) > 0 for r in eligible)
check("eligible_count", len(eligible), 650)
check("positive_penalty_count", positive, 650)
report = {
    "status": "PASS" if not failures else "FAIL", "identity_checks": count-len(fd_errors),
    "finite_difference_checks": len(fd_errors), "max_identity_absolute_error": max(errors),
    "max_finite_difference_absolute_error": max(fd_errors), "failures": failures,
    "eligible_incorrect_rollouts": len(eligible), "strictly_positive_penalties": positive,
    "seed": 20261126, "identity_atol": 1e-10, "identity_rtol": 1e-8,
    "finite_difference_atol": 1e-7, "finite_difference_rtol": 1e-6,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "amendment_sha256": hashlib.sha256((ROOT/"AMENDMENT.md").read_bytes()).hexdigest(),
    "scope": "Post-result mathematical validation; no model gradient or new training result."
}
(ROOT/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
(ROOT/"SHA256SUMS.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+
    "  "+p.name+"\n" for p in sorted(ROOT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"))
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures))
