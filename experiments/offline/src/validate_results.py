"""Independent recomputation from saved rollout output and source ledger."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr
from audit import Checks, load, read_jsonl, require, sha, verify_seal, write


def validate(run,output,repeat=None):
    run,output=Path(run),Path(output);verify_seal(run);ck=Checks()
    ledger=read_jsonl(run/"joined_rollouts.jsonl"); token_rows=read_jsonl(run/"token_records.jsonl")
    index={(r["group_id"],r["sample_idx"]):r for r in ledger}
    require(len(index)==4000==len(token_rows),"Source ledger completeness")
    for t in token_rows:
        r=index[(t["group_id"],t["sample_idx"])]; ids=t["token_ids"]
        require(len(ids)==r["stored_generated_token_count"],"Token count mismatch")
        require(hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()==r["token_sequence_sha256"],"Token hash mismatch")
    cfg=load(run/"config.json"); cal=load(output/"calibration.json")
    group_src=defaultdict(list)
    for r in ledger: group_src[r["group_id"]].append(r)
    numerator=math.fsum(r["learned_penalty"] for r in ledger if r["group_type"]=="mixed" and not r["correctness"])
    denom=sum(r["stored_generated_token_count"] for r in ledger if r["group_type"]=="mixed" and not r["correctness"])
    ck.close("independent_calibration",cal["coefficient"],numerator/denom)
    output_rows=read_jsonl(output/"rollout_results.jsonl"); seen=set(); groups=defaultdict(list)
    require(len(output_rows)==32000,"Expected four rules x two normalizations x 4000 records")
    for row in output_rows:
        key=(row["group_id"],row["sample_idx"],row["rule"],row["normalization"])
        require(key not in seen,"Duplicate result row");seen.add(key)
        source=index[key[:2]];wrong=[x for x in group_src[key[0]] if not x["correctness"]]
        expected=0.
        if source["group_type"]=="mixed" and not source["correctness"]:
            if row["rule"]=="learned": expected=.1*source["deployed_distance"]
            elif row["rule"]=="constant": expected=statistics.mean(.1*x["deployed_distance"] for x in wrong)
            elif row["rule"]=="length_global": expected=cal["coefficient"]*source["stored_generated_token_count"]
            elif row["rule"]=="length_group": expected=statistics.mean(.1*x["deployed_distance"] for x in wrong)*source["stored_generated_token_count"]/statistics.mean(x["stored_generated_token_count"] for x in wrong)
            else: raise ValueError("Unknown rule")
        ck.close("saved_penalty",row["penalty"],expected);ck.close("saved_R",row["R"],source["correctness"])
        ck.close("saved_reward",row["shaped_reward"],source["correctness"]-expected)
        groups[(row["rule"],row["normalization"],row["group_id"])].append(row)
    global_vectors=defaultdict(lambda:[[],[]]); pergroup=defaultdict(list); recomputed_group={}
    def independent(a,b):
        aa=math.fsum(x*x for x in a);bb=math.fsum(x*x for x in b)
        if aa==0 or bb==0:return None
        dot=math.fsum(x*y for x,y in zip(a,b));c=dot/math.sqrt(aa*bb)
        return {"cosine":c,"energy":max(0.,1-c*c),"norm_ratio":math.sqrt(bb/aa),"relative_change":math.sqrt(math.fsum((y-x)**2 for x,y in zip(a,b))/aa)}
    for (rule,norm,gid),rows in sorted(groups.items()):
        rows.sort(key=lambda x:x["sample_idx"]);require(len(rows)==8,"Lost group sample")
        rewards=[x["R"] for x in rows];shaped=[x["shaped_reward"] for x in rows]
        a=[x-statistics.mean(rewards) for x in rewards];b=[x-statistics.mean(shaped) for x in shaped]
        if norm=="std":
            a=[x/(statistics.pstdev(rewards)+cfg["epsilon"]) for x in a]
            b=[x/(statistics.pstdev(shaped)+cfg["epsilon"]) for x in b]
        ck.close("saved_binary_advantage",[x["binary_advantage"] for x in rows],a)
        ck.close("saved_shaped_advantage",[x["shaped_advantage"] for x in rows],b)
        global_vectors[(rule,norm)][0].extend(a);global_vectors[(rule,norm)][1].extend(b)
        m=independent(a,b);recomputed_group[(rule,norm,gid)]=m
        if m: pergroup[(rule,norm)].append(m)
    summaries=load(output/"summary.json");sens=load(output/"sensitivity_results.json")
    for (rule,norm),(a,b) in global_vectors.items():
        target=(sens["length_group"] if rule=="length_group" else summaries["rules"][rule])["normalizations"][norm]
        m=independent(a,b)
        for key,value in m.items():ck.close("independent_global_"+key,value,target["global"][key])
        for key in m:
            values=[x[key] for x in pergroup[(rule,norm)]];q=np.quantile(values,[0,.25,.5,.75,1],method="linear")
            ck.close("independent_per_group_"+key,q,[target["per_group"][key][n] for n in ("minimum","q1","median","q3","maximum")])
        gtarget=target["global"];ck.close("saved_energy_partition",gtarget["within_energy"]+gtarget["weight_energy"],gtarget["energy"])
    with (output/"group_metrics.csv").open(encoding="utf-8",newline="") as f:
        csvrows=list(csv.DictReader(f))
    require(len(csvrows)==4000,"Group metric CSV completeness")
    for row in csvrows:
        target=recomputed_group[(row["rule"],row["normalization"],row["group_id"])]
        if target:
            for k,v in target.items():ck.close("saved_group_csv_"+k,float(row[k]),v)
        else:require(all(row[k]=="" for k in ("cosine","energy","norm_ratio","relative_change")),"Undefined group metric serialized as number")
    with (output/"length_association.csv").open(encoding="utf-8",newline="") as f: assoc=list(csv.DictReader(f))
    require(len(assoc)==185,"Association completeness")
    for row in assoc:
        wrong=[r for r in group_src[row["group_id"]] if not r["correctness"]]
        d=[r["deployed_distance"] for r in wrong];l=[r["stored_generated_token_count"] for r in wrong]
        if len(d)<2 or len(set(d))==1 or len(set(l))==1: require(row["spearman"]=="" and row["undefined_reason"],"Undefined association mishandled")
        else: ck.close("independent_scipy_spearman",float(row["spearman"]),float(spearmanr(d,l).statistic))
    hist={str(m):sum(g[0]["group_type"]=="mixed" and sum(1-r["correctness"] for r in g)==m for g in group_src.values()) for m in range(1,8)}
    require(hist==summaries["structure"]["mixed_group_incorrect_histogram"],"Structure histogram mismatch")
    require(sum(hist.values())==185,"Mixed denominator mismatch")
    repetition={"performed":False}
    if repeat:
        repeat=Path(repeat); names=sorted(p.name for p in output.iterdir() if p.suffix in (".json",".jsonl",".csv") and p.name!="validation.json")
        # Only deterministic analysis outputs are compared, not later report/claim artifacts.
        names=[n for n in names if (repeat/n).exists()]
        require(len(names)==8,"Unexpected deterministic result file count")
        values={n:sha(output/n)==sha(repeat/n) for n in names}
        require(all(values.values()),"Deterministic repeat differs")
        repetition={"performed":True,"byte_identical":True,"files":values}
    result={"schema_version":1,"passed":True,"independent_saved_output_checks":ck.results,"sealed_source_and_input_hashes_verified":True,"token_records_checked":4000,"rollout_results_checked":len(output_rows),"group_metric_rows_checked":len(csvrows),"association_rows_checked":len(assoc),"deterministic_repeat":repetition}
    write(output/"validation.json",result)
    print("Independent saved-output validation passed; deterministic repeat:",repetition["performed"])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--repeat",type=Path)
    a=p.parse_args();validate(a.run,a.output,a.repeat)
