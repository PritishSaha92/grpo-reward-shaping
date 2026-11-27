"""Independent checks using the sealed original implementation, plus empirical plots."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from analyze import load, rows, write, check, sha, Checks

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();run=a.run
    original=Path(load(run/"config.json")["original_run"])
    spec=importlib.util.spec_from_file_location("old_audit",original/"src/audit.py")
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=Checks();c=load(run/"inputs/calibration.json")["coefficient"]
    data=rows(run/"inputs/ledger.jsonl");by={}
    for row in data: by.setdefault(row["group_id"],[]).append(row)
    summary=load(run/"results/summary.json")
    for convention in ("std","mean"):
        bs={"learned":[],"length_global":[]};ws={k:[] for k in bs}
        for gid,rr in sorted(by.items()):
            rr=sorted(rr,key=lambda x:x["sample_idx"]);r=np.array([d["correctness"] for d in rr],float)
            if not 0<r.sum()<8: continue
            aa=old.normalize(r,convention)
            for rule in bs:
                q=old.penalties(r,[d["deployed_distance"] for d in rr],[d["stored_generated_token_count"] for d in rr],rule,c)
                b=old.normalize(r-q,convention);w=b-aa*np.dot(aa,b)/np.dot(aa,aa)
                bs[rule].extend(b);ws[rule].extend(w)
        for key,pool in (("global_advantage_comparison",bs),("global_residual_comparison",ws)):
            computed=old.geometry(pool["learned"],pool["length_global"])
            target=summary["normalizations"][convention][key]
            for k,kold in (("c","cosine"),("rho","norm_ratio"),("D","relative_change"),("E","energy")):
                checks.close("independent_"+key+"_"+k,computed[kold],target[k])
    oldtable=list(csv.DictReader((original/"results/tables/full_geometry.csv").open()))
    table=list(csv.DictReader((run/"results/geometry.csv").open()))
    for new in table:
        rule="length_global" if new["rule"]=="length" else new["rule"]
        previous=next(x for x in oldtable if x["rule"]==rule and x["normalization"]==new["normalization"])
        for k,ko in (("c","cosine"),("rho","norm_ratio"),("D","relative_change"),("E","energy"),("E_within","within_energy"),("E_weight","weight_energy")):
            checks.close("original_result_"+k,float(new[k]),float(previous[ko]))
    groupdata=list(csv.DictReader((run/"results/per_group.csv").open()))
    std=[x for x in groupdata if x["normalization"]=="std"]
    mean=[x for x in groupdata if x["normalization"]=="mean"]
    for x,y in zip(std,mean,strict=True):
        check(x["group_id"]==y["group_id"],"Group order")
        for k in ("learned_energy","length_energy","advantage_cosine","residual_cosine"):
            if x[k]: checks.close("normalization_invariant_per_group_"+k,float(x[k]),float(y[k]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    out=run/"results/figures";out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharex=True,sharey=True,layout="constrained")
    for ax,(title,cohort) in zip(axes,[("All mixed groups (n=185)",std),("At least two incorrect (n=132)",[x for x in std if x["opportunity"]=="True"])],strict=True):
        for rule,color in (("learned","#19776a"),("length","#bc663d")):
            vals=np.sort([100*float(x[rule+"_energy"]) for x in cohort])
            ax.step(np.r_[0,vals,50],np.r_[0,np.arange(1,len(vals)+1)/len(vals),1],where="post",color=color,label=rule.capitalize())
        ax.set(title=title,xlabel="Within-group orthogonal energy (%)",xlim=(0,50),ylim=(0,1.02))
        ax.grid(alpha=.15);ax.legend(loc="lower right",frameon=False)
    axes[0].set_ylabel("Fraction of groups at or below")
    for ext in ("pdf","svg","png"): fig.savefig(out/("conditional_energy."+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
    x=[100*float(d["learned_energy"]) for d in std if d["opportunity"]=="True"]
    y=[100*float(d["length_energy"]) for d in std if d["opportunity"]=="True"]
    axes[0].scatter(x,y,s=16,alpha=.6,color="#19776a");axes[0].plot([0,50],[0,50],"--",color="0.6",lw=1)
    axes[0].set(xlabel="Learned energy (%)",ylabel="Length energy (%)",xlim=(0,50),ylim=(0,50),title="132 opportunity groups")
    values=np.sort([float(d["residual_cosine"]) for d in std if d["residual_cosine"]])
    axes[1].step(np.r_[-1,values,1],np.r_[0,np.arange(1,len(values)+1)/len(values),1],where="post",color="#19776a")
    axes[1].axvline(0,color="0.6",ls="--",lw=1)
    axes[1].set(xlabel="Learned-length residual cosine",ylabel="Fraction of defined groups at or below",xlim=(-1,1),ylim=(0,1.02),title="131 defined residual comparisons")
    for ax in axes: ax.grid(alpha=.15)
    for ext in ("pdf","svg","png"): fig.savefig(out/("direct_residual_comparison."+ext),dpi=180)
    plt.close(fig)
    write(run/"results/independent_validation.json",dict(status="PASS",checks=checks.items,
        comparison_source_sha256=sha(original/"src/audit.py"),
        note="Independent original normalizer and explicit projection; not an independent data sample"))
    print(json.dumps(dict(status="PASS",checks=sum(v["count"] for v in checks.items.values()),figures=str(out))))

if __name__=="__main__": main()
