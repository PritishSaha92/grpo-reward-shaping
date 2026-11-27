"""Post-hoc assignment comparison. Run prepare, then the sealed copy's run command."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_LEDGER = "c74f8705d3df940ea7f697d36747d2525ee187152ed709a5ca3e54a6a223a4a0"
EPS = 1e-4
SEED, SHUFFLES = 20261126, 32
RULES = ("learned", "constant", "length")
ATOL, RTOL = 1e-10, 1e-8


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines() if s]
def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(p, data):
    with Path(p).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def check(ok, reason):
    if not ok: raise ValueError(reason)


class Checks:
    def __init__(self): self.items = {}
    def close(self, name, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        item = self.items.setdefault(name, {"count":0,"max_absolute_error":0.})
        item["count"] += 1
        item["max_absolute_error"] = max(error, item["max_absolute_error"])
        check(np.allclose(a,b,atol=ATOL,rtol=RTOL), f"{name}: {error}")


def normed(r, convention):
    x = np.asarray(r, dtype=np.float64); x = x - x.mean()
    return x if convention == "mean" else x / (np.sqrt(np.mean(x*x)) + EPS)


def geom(a, b):
    a,b = np.asarray(a),np.asarray(b); aa,bb = float(a@a),float(b@b)
    if aa == 0 or bb == 0:
        return dict(c=None,rho=None,D=None,E=None)
    c = float(np.clip((a@b)/np.sqrt(aa*bb), -1, 1))
    return dict(c=c,rho=float(np.sqrt(bb/aa)),D=float(np.linalg.norm(b-a)/np.sqrt(aa)),E=max(0.,1-c*c))


def decompose(r, a, b, checks=None):
    alpha = float(a@b/(a@a))
    w = b.copy()
    for label in (0,1):
        mask = r == label; w[mask] -= w[mask].mean()
    if checks:
        checks.close("class_centering_equals_projection", w, b-alpha*a)
        checks.close("residual_orthogonality", a@w, 0.)
    if np.linalg.norm(w) <= 1e-12 * max(1.,float(np.linalg.norm(b))): w[:] = 0.
    return alpha,w


def pooled(a, b, residual):
    g = geom(a,b); alpha = float(a@b/(a@a))
    within = float(residual@residual/(b@b))
    weighting = float(np.sum((b-residual-alpha*a)**2)/(b@b))
    return dict(**g,E_within=within,E_weight=weighting)


def describe(values):
    good = [x for x in values if x is not None]
    out = dict(defined=len(good),excluded=len(values)-len(good))
    return dict(**out, **dict(zip(("min","q1","median","q3","max"),
        map(float,np.quantile(good,[0,.25,.5,.75,1],method="linear")) if good else [None]*5)))


def verify_seal(run):
    seal = load(run/"seal.json")
    for rel, expected in seal["files"].items(): check(sha(run/rel)==expected, "Sealed input changed: "+rel)
    return len(seal["files"])


def synthetic():
    c=Checks(); r=np.array([1.,1.,0.,0.]); a=normed(r,"std")
    b=normed(r-np.array([0.,0.,1.,3.]),"std")
    c.close("toy_cosine_squared",geom(a,b)["c"]**2,9/11)
    c.close("constant_direction",geom(a,normed(r-[0,0,2,2],"std"))["c"],1.)
    _,w=decompose(r,a,b,c); perm=np.array([0,1,3,2])
    c.close("toy_permutation_geometry",list(geom(a,b).values()),list(geom(a,b[perm]).values()))
    h=np.array([[1.,0.],[0.,1.],[1.,2.],[-2.,1.]])
    check(np.linalg.norm(b@h-b[perm]@h)>0.1,"Synthetic assignment must matter")
    hc=h.copy()
    for label in (0,1): hc[r==label]-=hc[r==label].mean(axis=0)
    c.close("within_class_gradient_contrast",w@h,w@hc)
    c.close("homogeneous_advantage",normed(np.zeros(8),"std"),np.zeros(8))
    rs=np.array([1.,1.,1.,0.]); aa=normed(rs,"std");bb=normed(rs-[0,0,0,2.],"std")
    c.close("single_wrong_zero_residual",decompose(rs,aa,bb)[1],np.zeros(4))
    rng=np.random.Generator(np.random.PCG64(42))
    for n in (4,8,16):
        rr=np.r_[np.ones(n//2),np.zeros(n-n//2)]
        for convention in ("std","mean"):
            aa=normed(rr,convention)
            for _ in range(20):
                q=(1-rr)*rng.random(n)*3;bb=normed(rr-q,convention)
                al,ww=decompose(rr,aa,bb,c)
                hh=rng.normal(size=(n,11))
                c.close("gradient_decomposition",bb@hh,al*(aa@hh)+ww@hh)
    return dict(status="PASS",checks=c.items)


def prepare(original, pilot, destination):
    check(not destination.exists(),"Refusing existing run directory")
    verify_seal(original)
    check(sha(original/"joined_rollouts.jsonl")==EXPECTED_LEDGER,"Wrong ledger")
    destination.mkdir(parents=True)
    for rel,src in {
        "inputs/ledger.jsonl": original/"joined_rollouts.jsonl",
        "inputs/calibration.json":original/"results/calibration.json",
        "inputs/historical_pilot.json":pilot,
        "PROTOCOL.md":HERE/"PROTOCOL.md",
        "src/analyze.py":Path(__file__),
    }.items():
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    write(destination/"synthetic_validation.json",synthetic())
    write(destination/"config.json",dict(seed=SEED,shuffles=SHUFFLES,epsilon=EPS,
        normalizations=["std","mean"],primary="std",zero_residual_relative_threshold=1e-12,
        original_run=str(original.resolve()),original_seal_sha256=sha(original/"seal.json"),
        prior_knowledge="Original CPU audit and validation amendments known; new assignment comparisons not inspected",
        post_hoc=True))
    write(destination/"environment.json",dict(python=sys.version,numpy=importlib.metadata.version("numpy")))
    files={str(p.relative_to(destination)).replace("\\","/"):sha(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write(destination/"seal.json",dict(schema_version=1,sealed_at_utc=datetime.now(timezone.utc).isoformat(),files=files))
    print(json.dumps(dict(prepared=str(destination),seal_sha256=sha(destination/"seal.json"))))


def analyze(run):
    verify_seal(run)
    out=run/"results";check(not out.exists(),"Refusing overwrite of results");out.mkdir()
    config=load(run/"config.json");original=Path(config["original_run"])
    if not original.is_absolute(): original=(run/original).resolve()
    check(sha(original/"seal.json")==config["original_seal_sha256"],"Original seal changed")
    verify_seal(original)
    c=Checks(); by=defaultdict(list)
    for row in rows(run/"inputs/ledger.jsonl"): by[row["group_id"]].append(row)
    check(len(by)==500,"500 groups required")
    groups=[];counts=Counter()
    coefficient=load(run/"inputs/calibration.json")["coefficient"]
    for gid in sorted(by):
        data=sorted(by[gid],key=lambda x:x["sample_idx"])
        check([d["sample_idx"] for d in data]==list(range(8)),"Bad sample indices")
        r=np.array([d["correctness"] for d in data],dtype=float)
        check(np.all((r==0)|(r==1)),"Nonbinary correctness")
        counts["all_wrong" if r.sum()==0 else "all_correct" if r.sum()==8 else "mixed"]+=1
        if not 0<r.sum()<8: continue
        q=np.array([d["learned_penalty"] for d in data]);mask=r==0
        c.close("stored_learned_penalty",q,.1*np.array([d["deployed_distance"] for d in data]))
        check(np.all(q[mask]>0) and np.all(q[~mask]==0),"Penalty eligibility changed")
        const=np.where(mask,q[mask].mean(),0.)
        length=np.where(mask,coefficient*np.array([d["stored_generated_token_count"] for d in data]),0.)
        groups.append(dict(gid=gid,r=r,learned=q,constant=const,length=length,n_wrong=int(mask.sum())))
    check(counts==dict(all_wrong=47,mixed=185,all_correct=268),"Cohort mismatch")
    rng=np.random.Generator(np.random.PCG64(SEED)); maps={}
    for group in groups:
        indices=np.flatnonzero(group["r"]==0);perms=[]
        for j in range(SHUFFLES):
            p=np.arange(8);p[indices]=rng.permutation(indices);perms.append(p.tolist())
        maps[group["gid"]]=perms
    write(out/"permutations.json",dict(seed=SEED,rng="numpy.PCG64",meaning="destination index -> source index",mappings=maps))
    pergroup=[];tables=[];summaries={};weight_plan={};shuffle_checks=[]
    for convention in ("std","mean"):
        arrays={k:[] for k in ("binary",)+RULES};residuals={k:[] for k in RULES};local=[]
        for group in groups:
            gid,r=group["gid"],group["r"];a=normed(r,convention);arrays["binary"].append(a)
            bs={};ws={};alphas={};item=dict(normalization=convention,group_id=gid,n_wrong=group["n_wrong"],opportunity=group["n_wrong"]>=2)
            for rule in RULES:
                b=normed(r-group[rule],convention);alpha,w=decompose(r,a,b,c)
                bs[rule],ws[rule],alphas[rule]=b,w,alpha
                arrays[rule].append(b);residuals[rule].append(w)
                item[rule+"_energy"]=float(w@w/(b@b))
            pair=geom(bs["learned"],bs["length"])
            wr=geom(ws["learned"],ws["length"])
            item.update(advantage_cosine=pair["c"],advantage_relative_difference=pair["D"],
                residual_cosine=wr["c"],residual_norm_learned=float(np.linalg.norm(ws["learned"])),
                residual_norm_length=float(np.linalg.norm(ws["length"])),
                residual_undefined_reason=("both_zero" if not np.any(ws["learned"]) and not np.any(ws["length"]) else "learned_zero" if not np.any(ws["learned"]) else "length_zero" if not np.any(ws["length"]) else None),
                energy_difference=item["learned_energy"]-item["length_energy"])
            local.append(item);pergroup.append(item)
            if convention=="std":
                weight_plan[gid]=dict(group_id=gid,correctness=r.tolist(),binary=a.tolist(),
                    shaped={k:bs[k].tolist() for k in RULES},residual={k:ws[k].tolist() for k in ("learned","length")},
                    alpha=alphas,permutations=maps[gid])
        arrays={k:np.concatenate(v) for k,v in arrays.items()};residuals={k:np.concatenate(v) for k,v in residuals.items()}
        baseline={}
        for rule in RULES:
            values=pooled(arrays["binary"],arrays[rule],residuals[rule]);baseline[rule]=values
            c.close("pooled_energy_partition",values["E"],values["E_within"]+values["E_weight"])
            tables.append(dict(normalization=convention,rule=rule,**values))
        global_pair=geom(arrays["learned"],arrays["length"])
        residual_pair=geom(residuals["learned"],residuals["length"])
        summary=dict(global_advantage_comparison=global_pair,global_residual_comparison=residual_pair,
            residual_norms={k:float(np.linalg.norm(residuals[k])) for k in ("learned","length")},cohorts={})
        fields=("learned_energy","length_energy","energy_difference","advantage_cosine","residual_cosine")
        for label,cohort in (("all_mixed",local),("at_least_two_incorrect",[d for d in local if d["opportunity"]])):
            summary["cohorts"][label]=dict(n_groups=len(cohort),**{k:describe([d[k] for d in cohort]) for k in fields},
                residual_exclusions=dict(Counter(d["residual_undefined_reason"] for d in cohort if d["residual_undefined_reason"])))
        summaries[convention]=summary
        for j in range(SHUFFLES):
            bb=[];ww=[];changed=0
            for group in groups:
                r=group["r"];a=normed(r,convention);b=normed(r-group["learned"],convention)
                perm=np.array(maps[group["gid"]][j]);bp=normed(r-group["learned"][perm],convention)
                al,w=decompose(r,a,b);alp,wp=decompose(r,a,bp,c)
                c.close("permuted_advantages",bp,b[perm]);c.close("permuted_residuals",wp,w[perm]);c.close("permuted_alpha",al,alp)
                c.close("local_permutation_geometry",list(geom(a,b).values()),list(geom(a,bp).values()))
                bb.append(bp);ww.append(wp);changed+=int(not np.array_equal(group["learned"],group["learned"][perm]))
            values=pooled(arrays["binary"],np.concatenate(bb),np.concatenate(ww))
            c.close("pooled_permutation_geometry",list(values.values()),list(baseline["learned"].values()))
            shuffle_checks.append(dict(normalization=convention,shuffle=j,changed_groups=changed,**values))
    # Pin the old pilot's complete sample, including structurally inactive cases.
    ids=load(run/"inputs/historical_pilot.json")["metadata"]["selected_group_ids"]
    check(len(ids)==len(set(ids))==32 and all(i in weight_plan for i in ids),"Pilot selection mismatch")
    plan=dict(normalization="std",epsilon=EPS,groups=[weight_plan[i] for i in ids],
        selection="Same 32 IDs and order as recorded pilot; no new outcome-based selection",
        n_groups=32,n_rollouts=256,n_permutable_groups=sum(sum(x==0 for x in weight_plan[i]["correctness"])>=2 for i in ids))
    write(out/"gpu_weight_plan.json",plan)
    count_op=sum(g["n_wrong"]>=2 for g in groups)
    summary=dict(scope="Post-hoc fixed-sample descriptive comparison; no retraining or model gradients",counts=dict(counts),
        opportunity=dict(n_groups=count_op,fraction_of_all=count_op/500,fraction_of_mixed=count_op/185,
            single_incorrect_groups=185-count_op),normalizations=summaries,permutation_draws=SHUFFLES,
        gpu_sample=dict(n_groups=32,n_rollouts=256,n_permutable_groups=plan["n_permutable_groups"]))
    write(out/"summary.json",summary);csv_write(out/"per_group.csv",pergroup)
    csv_write(out/"geometry.csv",tables);csv_write(out/"permutation_checks.csv",shuffle_checks)
    write(out/"validation.json",dict(status="PASS",checks=c.items,
        original_seal_verified_files=verify_seal(original),extension_seal_verified_files=verify_seal(run)))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("synthetic")
    s=sub.add_parser("prepare");s.add_argument("--original",type=Path,required=True);s.add_argument("--pilot",type=Path,required=True);s.add_argument("--destination",type=Path,required=True)
    s=sub.add_parser("run");s.add_argument("--run",type=Path,required=True)
    a=p.parse_args()
    if a.command=="synthetic": print(json.dumps(synthetic(),indent=2))
    elif a.command=="prepare": prepare(a.original,a.pilot,a.destination)
    else: analyze(a.run)
