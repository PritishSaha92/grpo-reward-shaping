"""CPU-only sealed counterfactual audit. No training, inference or network calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata

ATOL, RTOL = 1e-10, 1e-8
RULES = ("learned", "constant", "length_global", "length_group")
NORMS = ("std", "mean")
LABELS = {"learned": "Learned-distance shaping", "constant": "Group-constant penalty",
          "length_global": "Token-length shaping", "length_group": "Per-group matched length"}
RAW = "LaViDA/eval_results/oracle_phase3_seed0_n8/g7_principled_cr_seed0_math500_n8_redecode/math500_pass8_results.json"
DECODER = RAW.replace("math500_pass8_results.json", "config.json")
GEOM = "grpo-mechanism-audit/evidence/inputs/geometry_records.json"
TOKENIZER = "LaViDA/outputs/base_arm/oracle_matrix/oracleaug-probe-b/lavida-variant-B-seed0-alpha0p05/lavida-variant-B-seed0-alpha0p05/tokenizer.json"
EXPECTED = {RAW: "d5a29873bede905e4501a365677793c65325e032aec6f426762053a003cc27bd",
            DECODER: "da2377b9e8d4b3bdd0d2a52405dc676aff5454be9c01a07fff572eb9436dc966",
            GEOM: "ea070a0281d6dab2d9f3a08745c2d0df19eccc4cea5f6506ee6e14af14e4742e"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def csv_write(path, rows):
    if not rows: raise ValueError("Refusing ambiguous empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def timestamp():
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "Asia_Kolkata": now.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat()}


def require(ok, msg):
    if not ok: raise ValueError(msg)


class Checks:
    def __init__(self): self.results = {}
    def close(self, name, actual, expected):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        r = self.results.setdefault(name, {"comparisons": 0, "max_absolute_error": 0.0, "passed": True})
        r["comparisons"] += 1; r["max_absolute_error"] = max(r["max_absolute_error"], err)
        require(np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=False), f"{name}: error {err}")


def checked_unit(value, low, high):
    require(math.isfinite(value), "Nonfinite metric")
    require(low - ATOL <= value <= high + ATOL, "Metric outside numerical tolerance")
    return float(np.clip(value, low, high))


def normalize(rewards, convention, epsilon=1e-4):
    r = np.asarray(rewards, dtype=np.float64); x = r - r.mean()
    if convention == "mean": return x
    require(convention == "std", "Unknown normalization")
    sd = float(r.std(ddof=0))
    return np.zeros_like(x) if sd == 0 else x / (sd + epsilon)


def geometry(a, b, checks=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    aa, bb = float(a @ a), float(b @ b)
    if aa == 0 or bb == 0:
        return {"cosine": None, "energy": None, "norm_ratio": None, "relative_change": None,
                "undefined_reason": "both_zero" if aa == bb == 0 else "zero_binary" if aa == 0 else "zero_shaped"}
    c = checked_unit(float(a @ b) / math.sqrt(aa * bb), -1, 1)
    e = checked_unit(1 - c*c, 0, 1); rho = math.sqrt(bb/aa)
    rel = float(np.linalg.norm(b-a) / math.sqrt(aa))
    if checks:
        resid = b - (float(a @ b)/aa)*a
        checks.close("direct_projection_energy", float(resid @ resid)/bb, e)
        checks.close("relative_change_identity", rel*rel, 1 + rho*rho - 2*rho*c)
    return {"cosine": c, "energy": e, "norm_ratio": rho, "relative_change": rel, "undefined_reason": None}


def describe(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not len(v): return {"defined": 0, "excluded": len(values), "minimum": None, "q1": None, "median": None, "q3": None, "maximum": None}
    q = np.quantile(v, [0, .25, .5, .75, 1], method="linear")
    return dict(defined=len(v), excluded=len(values)-len(v), **dict(zip(("minimum", "q1", "median", "q3", "maximum"), map(float, q))))


def eligible(r):
    r = np.asarray(r)
    return (r == 0) if 0 < r.sum() < len(r) else np.zeros(len(r), dtype=bool)


def penalties(r, d, lengths, rule, coefficient, beta=.1):
    mask = eligible(r); q = np.zeros(len(r), dtype=float)
    if not mask.any(): return q
    learned = beta * np.asarray(d, dtype=float)[mask]
    if rule == "learned": q[mask] = learned
    elif rule == "constant": q[mask] = learned.mean()
    elif rule == "length_global": q[mask] = coefficient * np.asarray(lengths)[mask]
    elif rule == "length_group": q[mask] = learned.mean()*np.asarray(lengths)[mask]/np.asarray(lengths)[mask].mean()
    else: raise ValueError(rule)
    return q


def associate(distance, lengths):
    d, l = np.asarray(distance), np.asarray(lengths)
    reason = "fewer_than_two" if len(d) < 2 else "both_constant" if len(set(d)) == len(set(l)) == 1 else "distance_constant" if len(set(d)) == 1 else "length_constant" if len(set(l)) == 1 else None
    value = None if reason else checked_unit(float(np.corrcoef(rankdata(d, method="average"), rankdata(l, method="average"))[0,1]), -1, 1)
    return value, reason


def strict_join(raw, geom, projected, config):
    raw_rows = raw["per_problem"]; projected_rows = projected["per_problem"]
    by_id = {}; proj = {}
    for p in projected_rows:
        require(p["unique_id"] not in proj, "Duplicate projected group")
        proj[p["unique_id"]] = p
    for p in raw_rows:
        gid = p["unique_id"]
        require(isinstance(gid, str) and gid and gid not in by_id, "Duplicate/invalid raw group")
        require(gid in proj, "Missing projected group")
        require(len(p["samples"]) == len(proj[gid]["samples"]) == config["group_size"], "Sample count")
        by_id[gid] = p
    require(set(by_id) == set(proj), "Projected group identity mismatch")
    gmap = {}
    for row in geom["records"]:
        require(type(row["sample_idx"]) is int, "Invalid geometry sample index")
        key = (row["group_id"], row["sample_idx"])
        require(key not in gmap, "Duplicate geometry key")
        gmap[key] = row
    require(len(by_id) == config["expected_groups"], "Unexpected group count")
    out = []; tokens = []
    for gid in sorted(by_id):
        group = by_id[gid]; samples = group["samples"]
        require(all(type(s["correct"]) is bool for s in samples), "Correctness must be boolean")
        nc = sum(s["correct"] for s in samples)
        require(type(group["n_correct"]) is int and nc == group["n_correct"], "Raw count mismatch")
        require(type(group["any_correct"]) is bool and bool(nc) == group["any_correct"], "Raw any-correct mismatch")
        kind = "mixed" if 0 < nc < len(samples) else "all_correct" if nc else "all_wrong"
        for idx, s in enumerate(samples):
            key = (gid, idx); require(key in gmap, "Missing geometry sample")
            g = gmap[key]; ps = proj[gid]["samples"][idx]
            require(type(g["correct"]) is bool and type(ps["correct"]) is bool, "Non-boolean linked correctness")
            require(g["correct"] == s["correct"] == ps["correct"], "Correctness mismatch")
            require(s["prediction"] == ps["prediction"], "Prediction mismatch")
            require(g["n_correct"] == nc, "Geometry count mismatch")
            require(g["group_type"] == ("zero_correct" if kind == "all_wrong" else kind), "Geometry kind mismatch")
            d = g.get("deployed_distance")
            require(type(d) in (float, int) and math.isfinite(d) and d >= 0, "Invalid deployed distance")
            active = kind == "mixed" and not s["correct"]
            require(active or d == 0, "Noneligible deployed distance is nonzero")
            ids = s.get("token_ids")
            require(isinstance(ids, list) and len(ids) > 0 and all(type(v) is int and v >= 0 for v in ids), "Invalid completion token list")
            tokhash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
            out.append({"group_id": gid, "sample_idx": idx, "correctness": int(s["correct"]), "group_type": kind,
                        "n_correct": nc, "stored_generated_token_count": len(ids), "token_sequence_sha256": tokhash,
                        "deployed_distance": float(d), "learned_penalty": config["beta"]*float(d) if active else 0.0,
                        "termination_status": "unknown", "nominal_cap_reached": None,
                        "effective_cap_reached": None, "effective_cap": None,
                        "cap_provenance": "Per-request effective limit and termination reason not retained"})
            tokens.append({"group_id": gid, "sample_idx": idx, "token_ids": ids})
    require(set(gmap) == {(x["group_id"],x["sample_idx"]) for x in out}, "Unexpected geometry records")
    return out, tokens


def prepare(workspace, run, config_path, source_dir):
    from tokenizers import Tokenizer
    workspace, run = Path(workspace), Path(run)
    require(not run.exists(), "Run path already exists; refuse overwrite")
    config = load(config_path)
    for rel, expected in EXPECTED.items(): require(sha(workspace/rel) == expected, "Hash mismatch: " + rel)
    raw, geom = load(workspace/RAW), load(workspace/GEOM)
    require(geom["metadata"]["source_results_sha256"] == EXPECTED[RAW], "Geometry source hash mismatch")
    require(geom["metadata"]["beta"] == config["beta"], "Beta mismatch")
    projection = workspace/"grpo-mechanism-audit/evidence/inputs/G7_redecode.json"
    rows, token_rows = strict_join(raw, geom, load(projection), config)
    decoder = load(workspace/DECODER)
    require(decoder["max_tokens"] == 3072 and decoder["max_model_len"] == 4096, "Decoder limits changed")
    for r in rows:
        require(r["stored_generated_token_count"] <= decoder["max_tokens"], "Stored length exceeds nominal limit")
        r["nominal_cap_reached"] = r["stored_generated_token_count"] == decoder["max_tokens"]
    t = Tokenizer.from_file(str(workspace/TOKENIZER))
    special = {x["id"]: x["content"] for x in load(workspace/TOKENIZER)["added_tokens"] if x.get("special")}
    decoded = Counter(); suffix = Counter(); terminal = Counter(); all_special = Counter()
    for p in raw["per_problem"]:
        for s in p["samples"]:
            require("finish_reason" not in s, "New termination metadata needs explicit review")
            text = t.decode(s["token_ids"], skip_special_tokens=True)
            if text == s["completion"]: decoded["exact"] += 1
            else:
                require(text.startswith(s["completion"]), "IDs do not decode to saved completion")
                tail = text[len(s["completion"]):]
                require(tail == "\n\n\nQuestion:", "Unexpected retained stop suffix")
                decoded["saved_text_plus_stop_suffix"] += 1; suffix[tail] += 1
            for v in s["token_ids"]:
                if v in special: all_special[special[v]] += 1
            if s["token_ids"][-1] in special: terminal[special[s["token_ids"][-1]]] += 1
    counts = Counter(r["group_type"] for r in rows if r["sample_idx"] == 0)
    require(counts == {"mixed":185,"all_wrong":47,"all_correct":268}, "Cohort mismatch")
    require(sum(r["group_type"] == "mixed" and r["correctness"] == 0 for r in rows) == 650, "Eligibility mismatch")
    run.mkdir(parents=True)
    shutil.copy2(config_path, run/"config.json")
    shutil.copy2(workspace/"grpo-mechanism-audit/MATH4AI_2027_CPU_EXPERIMENT_PROTOCOL.md", run/"protocol.md")
    (run/"src").mkdir()
    for p in sorted(Path(source_dir).glob("*.py")): shutil.copy2(p, run/"src"/p.name)
    (run/"reference").mkdir()
    refs = {"historical_metrics.json":"grpo-mechanism-audit/evidence/derived/coverage_advantage.json",
            "decoder_config.json":DECODER,
            "redecode_audit.json":RAW.replace("math500_pass8_results.json","redecode_audit.json"),
            "geometry_extractor.py":"LaViDA/scripts/extract_g7_coverage_advantage_records.py",
            "gradient_estimator.py":"grpo-mechanism-audit/evidence/estimators/estimate_g7_policy_gradient_alignment.py",
            "current_export_provenance.py":"LaViDA/lavida/eval/provenance.py"}
    for dest, src in refs.items(): shutil.copy2(workspace/src, run/"reference"/dest)
    jsonl(run/"joined_rollouts.jsonl", rows); jsonl(run/"token_records.jsonl", token_rows)
    manifest_files = list(EXPECTED) + [TOKENIZER, str(projection.relative_to(workspace)).replace("\\","/")] + list(refs.values())
    manifest = {"schema_version":1, "prepared_at":timestamp(), "files":[{"path":p,"sha256":sha(workspace/p)} for p in sorted(set(manifest_files))],
                "groups":len(counts) and len(raw["per_problem"]), "samples":len(rows), "group_counts":dict(counts),
                "token_semantics": {"basis":"Historical extraction and replay consumers treat token_ids as completion-only IDs; recorded redecode audit says IDs saved for latent extraction. All stored IDs crosschecked against saved completion text with archived local tokenizer.",
                 "exact_original_exporter":"Not retained; current exporter snapshot is supporting documentation, not the historical execution source",
                 "local_tokenizer_scope":"Local archived vocabulary used to verify stored-token semantics; not claimed to pin the historical model/tokenizer revision",
                 "decode_counts":dict(decoded), "retained_suffixes":dict(suffix), "terminal_special_counts":dict(terminal),"all_special_counts":dict(all_special),
                 "length_definition":config["length"], "effective_cap":None,"termination_status":"unknown for all samples; special tokens not treated as stop-reason evidence"},
                "extraction_limits": {k:v for k,v in geom["metadata"].items() if "hash" in k or k in ("base_model_revision","teacher_forcing_context_truncation")}}
    write(run/"input_manifest.json",manifest)
    versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","matplotlib","tokenizers","PyMuPDF")}
    commits={}
    for name in (".","LaViDA"):
        proc=subprocess.run(["git","-C",str(workspace/name),"rev-parse","HEAD"],capture_output=True,text=True)
        commits[name]=proc.stdout.strip() if proc.returncode == 0 else None
    write(run/"environment.json",{"schema_version":1,"created_at":timestamp(),"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"dependencies":versions,"git_revisions":commits,"code_identity":"Exact source snapshots and hashes in seal.json; Git revisions do not include these uncommitted files","command":sys.argv})
    print("Preflight complete: 500 joined groups, 4000 token records, source hashes verified. New comparison not run.")


def seal(run):
    run=Path(run); require(not (run/"seal.json").exists(),"Already sealed")
    require(load(run/"synthetic_validation.json")["passed"],"Synthetic checks missing/failing")
    files={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    write(run/"seal.json",{"schema_version":1,"sealed_at":timestamp(),"analysis_outcomes_inspected":False,"prior_knowledge":"Historical audit statistics and token/provenance preflight only; new rules, rank correlations and structural histogram not inspected", "files":files})
    print("Sealed pre-result snapshot:",sha(run/"seal.json"))


def verify_seal(run):
    run=Path(run); sealed=load(run/"seal.json")
    for rel, expected in sealed["files"].items(): require(sha(run/rel)==expected,"Sealed file changed: "+rel)
    for p in Path(__file__).resolve().parent.glob("*.py"):
        require(p.name in {Path(x).name for x in sealed["files"] if x.startswith("src/")}, "Unsealed source file")
        require(sha(p)==sealed["files"]["src/"+p.name],"Execution code differs from sealed code")


def group_rows(rows):
    groups=defaultdict(list)
    for row in rows: groups[row["group_id"]].append(row)
    return [sorted(groups[k],key=lambda r:r["sample_idx"]) for k in sorted(groups)]


def partition(a_blocks,b_blocks,checks):
    a=np.concatenate(a_blocks); b=np.concatenate(b_blocks); result=geometry(a,b,checks)
    aa=float(a@a); bb=float(b@b)
    if aa==0 or bb==0: return {**result,"within_energy":None,"weight_energy":None}
    alpha=float(a@b)/aa; within=0.; weighting=0.
    for x,y in zip(a_blocks,b_blocks):
        xx=float(x@x)
        if xx==0:
            checks.close("homogeneous_shaped_zero",y,np.zeros_like(y)); continue
        ap=float(x@y)/xx; w=y-ap*x
        within+=float(w@w); weighting+=(ap-alpha)**2*xx
    result.update(within_energy=within/bb,weight_energy=weighting/bb)
    checks.close("energy_partition",result["within_energy"]+result["weight_energy"],result["energy"])
    mask=np.concatenate([np.full(len(x),bool(float(x@x))) for x in a_blocks])
    checks.close("zero_block_invariance",[result[k] for k in ("cosine","energy","norm_ratio","relative_change")],
                 [geometry(a[mask],b[mask])[k] for k in ("cosine","energy","norm_ratio","relative_change")])
    return result


def comparison(groups,coefficient,config,checks,include_rollouts=True):
    summary={}; metrics=[]; rollouts=[]
    for rule in RULES:
        norm_results={}; group_cos={}; penalty_total=0.; penalty_nonzero=0
        for convention in NORMS:
            ab=[]; bb=[]; mixed_metrics=[]; penalty_total=0.; penalty_nonzero=0
            for group in groups:
                r=np.array([x["correctness"] for x in group],dtype=float)
                d=np.array([x["deployed_distance"] for x in group]); lengths=np.array([x["stored_generated_token_count"] for x in group])
                mask=eligible(r); q=penalties(r,d,lengths,rule,coefficient,config["beta"])
                shaped=r-q; a=normalize(r,convention,config["epsilon"]); b=normalize(shaped,convention,config["epsilon"])
                ab.append(a); bb.append(b); gm=geometry(a,b,checks); penalty_total+=float(q.sum()); penalty_nonzero+=int(np.count_nonzero(q>0))
                if mask.any():
                    meanq=float(q[mask].mean()); u=np.where(mask,q-meanq,0); x=r-r.mean(); s=1+meanq
                    checks.close("centered_decomposition",shaped-shaped.mean(),s*x-u)
                    checks.close("orthogonality",float(x@u),0.)
                    pred=s*s*float(x@x)/(s*s*float(x@x)+float(u@u))
                    checks.close("cosine_squared_formula",gm["cosine"]**2,pred)
                    learned=config["beta"]*d[mask]
                    if rule in ("constant","length_group"):
                        checks.close("per_group_penalty_mean",meanq,float(learned.mean()))
                        checks.close("per_group_shaped_mean",shaped.mean(),r.mean()-float(learned.sum())/len(r))
                    if rule=="constant" or int(mask.sum())==1:
                        checks.close("zero_within_incorrect_component",gm["energy"],0.)
                        expected=s if convention=="mean" else s*(r.std()+config["epsilon"])/(s*r.std()+config["epsilon"])
                        checks.close("constant_norm_ratio",gm["norm_ratio"],expected)
                    if convention=="std": group_cos[group[0]["group_id"]]=gm["cosine"]
                    else: checks.close("normalization_angle_invariance",gm["cosine"],group_cos[group[0]["group_id"]])
                    mixed_metrics.append(gm)
                metrics.append({"group_id":group[0]["group_id"],"rule":rule,"normalization":convention,"group_type":group[0]["group_type"],"n_correct":int(r.sum()),"n_incorrect":int(len(r)-r.sum()),"mean_incorrect_penalty":float(q[mask].mean()) if mask.any() else None,**gm})
                if include_rollouts:
                    for i,source in enumerate(group):
                        rollouts.append({"group_id":source["group_id"],"sample_idx":source["sample_idx"],"rule":rule,"normalization":convention,"scope":"primary" if rule in config["primary_rules"] else "sensitivity","R":float(r[i]),"penalty":float(q[i]),"shaped_reward":float(shaped[i]),"binary_advantage":float(a[i]),"shaped_advantage":float(b[i])})
            glob=partition(ab,bb,checks)
            if rule=="constant": checks.close("constant_global_within_energy",glob["within_energy"],0.)
            norm_results[convention]={"global":glob,"per_group":{k:describe([m[k] for m in mixed_metrics]) for k in ("cosine","energy","norm_ratio","relative_change")},"eligible_mixed_groups":len(mixed_metrics),"undefined_homogeneous_groups":len(groups)-len(mixed_metrics)}
        summary[rule]={"label":LABELS[rule],"scope":"primary" if rule in config["primary_rules"] else "sensitivity","total_penalty":penalty_total,"positive_penalty_rollouts":penalty_nonzero,"normalizations":norm_results}
    return summary,metrics,rollouts


def association(groups):
    rows=[]
    for group in groups:
        if group[0]["group_type"] != "mixed": continue
        wrong=[r for r in group if r["correctness"]==0]
        d=[r["deployed_distance"] for r in wrong]; l=[r["stored_generated_token_count"] for r in wrong]
        c,reason=associate(d,l); m=len(wrong)
        rows.append({"group_id":group[0]["group_id"],"n_incorrect":m,"distinct_distances":len(set(d)),"distinct_lengths":len(set(l)),"length_tied_pairs":sum(v*(v-1)//2 for v in Counter(l).values()),"distance_tied_pairs":sum(v*(v-1)//2 for v in Counter(d).values()),"possible_pairs":m*(m-1)//2,"spearman":c,"undefined_reason":reason})
    def stats(rr):
        return {"groups":len(rr),"correlation":describe([r["spearman"] for r in rr]),"exclusions":dict(Counter(r["undefined_reason"] for r in rr if r["undefined_reason"])),"defined_exactly_two_incorrect":sum(r["n_incorrect"]==2 and r["spearman"] is not None for r in rr),"positive":sum(r["spearman"] is not None and r["spearman"]>0 for r in rr),"negative":sum(r["spearman"] is not None and r["spearman"]<0 for r in rr),"zero":sum(r["spearman"]==0 for r in rr)}
    return rows,{"primary":stats(rows),"at_least_three_incorrect":stats([r for r in rows if r["n_incorrect"]>=3])}


def censored_groups(groups):
    return {g[0]["group_id"] for g in groups if any(r["termination_status"]=="known_length" or r["nominal_cap_reached"] is True or r["effective_cap_reached"] is True for r in g)}


def compute(run,output):
    run,output=Path(run),Path(output); verify_seal(run)
    require(not output.exists(),"Output exists; refuse overwrite")
    config=load(run/"config.json"); checks=Checks(); rows=read_jsonl(run/"joined_rollouts.jsonl"); groups=group_rows(rows)
    # Reproduce historical learned row before any new rule outcome.
    a=[]; b=[]; en=[]
    for g in groups:
        r=np.array([x["correctness"] for x in g],dtype=float); q=np.array([x["learned_penalty"] for x in g])
        aa=normalize(r,"std"); bb=normalize(r-q,"std"); a.append(aa); b.append(bb)
        if 0<r.sum()<len(r): en.append(geometry(aa,bb)["energy"])
    historic=geometry(np.concatenate(a),np.concatenate(b),checks); ref=load(run/"reference/historical_metrics.json")["advantage"]
    keys={"cosine":"cosine_shaped_vs_binary_all_rollouts","energy":"novel_advantage_energy_fraction_all_rollouts","norm_ratio":"norm_ratio_shaped_to_binary_all_rollouts","relative_change":"relative_delta_norm_all_rollouts"}
    for k,v in keys.items(): checks.close("historical_"+k,historic[k],ref[v])
    checks.close("historical_median",float(np.median(en)),ref["mixed_group_novel_energy_fraction_median"])
    print("Historical geometry reproduced against full-precision reference.",flush=True)
    el=[r for r in rows if r["group_type"]=="mixed" and not r["correctness"]]
    numerator=float(sum(r["learned_penalty"] for r in el)); denominator=sum(r["stored_generated_token_count"] for r in el)
    require(denominator>0,"Zero calibration denominator"); coefficient=numerator/denominator
    summary,metrics,rollouts=comparison(groups,coefficient,config,checks)
    checks.close("global_penalty_matching",summary["length_global"]["total_penalty"],summary["learned"]["total_penalty"])
    assoc_rows,assoc_summary=association(groups)
    histogram={str(m):sum(r["n_incorrect"]==m for r in assoc_rows) for m in range(1,8)}
    structure={"total_groups":len(groups),"total_rollouts":len(rows),"group_counts":dict(Counter(g[0]["group_type"] for g in groups)),"eligible_rollouts":len(el),"eligible_rollout_fraction":len(el)/len(rows),"mixed_group_incorrect_histogram":histogram,"exactly_one_incorrect_count":histogram["1"],"mixed_group_denominator":len(assoc_rows),"exactly_one_incorrect_fraction":histogram["1"]/len(assoc_rows),"at_least_two_incorrect_fraction":1-histogram["1"]/len(assoc_rows)}
    def count_subset(sub):
        return {"denominator":len(sub),"termination_status":dict(Counter(r["termination_status"] for r in sub)),"nominal_cap_reached":sum(r["nominal_cap_reached"] is True for r in sub),"effective_cap_reached":sum(r["effective_cap_reached"] is True for r in sub),"effective_cap_unknown":sum(r["effective_cap_reached"] is None for r in sub)}
    removed=censored_groups(groups); retained=[g for g in groups if g[0]["group_id"] not in removed]
    total_pairs=sum(r["possible_pairs"] for r in assoc_rows); ties=sum(r["length_tied_pairs"] for r in assoc_rows)
    censor={"all_rollouts":count_subset(rows),"eligible_incorrect_rollouts":count_subset(el),"mixed_groups_with_confirmed_or_cap_reaching_sample":sum(g[0]["group_id"] in removed for g in groups if g[0]["group_type"]=="mixed"),"mixed_group_denominator":len(assoc_rows),"mixed_groups_with_unknown_termination":sum(any(r["termination_status"]=="unknown" for r in g) for g in groups if g[0]["group_type"]=="mixed"),"length_tied_pairs":ties,"possible_incorrect_pairs":total_pairs,"tied_pair_fraction":ties/total_pairs if total_pairs else None,"source_scope":"Nominal cap 3072 verified from config; per-request effective cap and termination reasons unavailable. Unknown is not untruncated."}
    cap={"removed_group_ids":sorted(removed),"removed_groups":len(removed),"retained_groups":len(retained),"coefficient":coefficient,"refitted":False,"remaining_unknown_termination":sum(r["termination_status"]=="unknown" for g in retained for r in g)}
    if not removed: cap.update(status="not_triggered",reason="No observed nominal/verified-effective cap hits or confirmed length termination; all stopping causes remain unknown")
    elif not any(g[0]["group_type"]=="mixed" for g in retained): cap.update(status="unavailable",reason="No mixed groups retained")
    else:
        cap.update(status="completed",results=comparison(retained,coefficient,config,checks,False)[0])
    output.mkdir(parents=True)
    write(output/"calibration.json",{"schema_version":1,"coefficient":coefficient,"numerator_total_learned_penalty":numerator,"denominator_total_eligible_tokens":denominator,"eligible_rollouts":len(el),"mean_learned_penalty":numerator/len(el),"unit":"penalty per stored generated token","calibrated_on":"full fixed post-hoc sample; reused across both normalizations and cap sensitivity"})
    write(output/"summary.json",{"schema_version":1,"rules":{r:summary[r] for r in config["primary_rules"]},"structure":structure,"length_association":assoc_summary,"historical_reproduction":historic,"interpretation":"Reconstructed fixed-sample advantage geometry; no new training, gradients or performance estimates"})
    write(output/"sensitivity_results.json",{"schema_version":1,"length_group":summary["length_group"],"cap_exclusion":cap})
    write(output/"censoring_report.json",censor); csv_write(output/"group_metrics.csv",metrics); csv_write(output/"length_association.csv",assoc_rows); jsonl(output/"rollout_results.jsonl",rollouts)
    write(output/"calculation_checks.json",{"passed":True,"checks":checks.results,"atol":ATOL,"rtol":RTOL})
    print("Completed primary 3x2 comparison, group-length sensitivity, structure, association and cap checks.",flush=True)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="action",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--workspace",type=Path,required=True); prep.add_argument("--run",type=Path,required=True); prep.add_argument("--config",type=Path,required=True)
    for name in ("seal","run"):
        q=sub.add_parser(name); q.add_argument("--run",type=Path,required=True)
        if name=="run": q.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.action=="prepare": prepare(args.workspace,args.run,args.config,Path(__file__).parent)
    elif args.action=="seal": seal(args.run)
    else: compute(args.run,args.output)


if __name__=="__main__": main()
