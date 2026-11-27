"""Render the completed CPU audit from checked machine-readable outputs."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit import LABELS, NORMS, RULES, load, write, sha, csv_write, require, verify_seal


def fmt(x,places=4):
    if x is None: return "undefined"
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP))


def percent(x): return fmt(100*x,2)


def render(run,output):
    run,output=Path(run),Path(output);verify_seal(run)
    require(load(output/"validation.json")["passed"],"Validation must pass before reporting")
    s=load(output/"summary.json");sen=load(output/"sensitivity_results.json");c=load(output/"calibration.json");cap=load(output/"censoring_report.json")
    rules={**s["rules"],"length_group":sen["length_group"]};structure=s["structure"];assoc=s["length_association"]
    tables=output/"tables";figures=output/"figures";tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    records=[]
    for rule in RULES:
        for norm in NORMS:
            data=rules[rule]["normalizations"][norm];g=data["global"]
            records.append({"rule":rule,"label":LABELS[rule],"scope":rules[rule]["scope"],"normalization":norm,**g,"median_group_energy":data["per_group"]["energy"]["median"],"mixed_groups":data["eligible_mixed_groups"]})
    csv_write(tables/"full_geometry.csv",records)
    table=["| Rule | Std cosine | Std norm ratio | Mean-only cosine | Mean-only norm ratio | Median group energy |","|---|---:|---:|---:|---:|---:|"]
    latex=[r"\begin{tabular}{lrrrrr}",r"\toprule",r"Rule & $c_{\mathrm{std}}$ & $\rho_{\mathrm{std}}$ & $c_{\mathrm{mean}}$ & $\rho_{\mathrm{mean}}$ & Median $e_p$ (\%) \\",r"\midrule"]
    for rule in s["rules"]:
        a=rules[rule]["normalizations"]["std"];b=rules[rule]["normalizations"]["mean"]
        vals=[fmt(a["global"]["cosine"]),fmt(a["global"]["norm_ratio"],6),fmt(b["global"]["cosine"]),fmt(b["global"]["norm_ratio"],6),percent(a["per_group"]["energy"]["median"])]
        table.append("| "+" | ".join([LABELS[rule],*vals[:-1],vals[-1]+"%"])+" |")
        latex.append(LABELS[rule]+" & "+" & ".join(vals)+r" \\")
    latex.extend([r"\bottomrule",r"\end{tabular}"])
    (tables/"primary_comparison.tex").write_text("\n".join(latex)+"\n",encoding="utf-8")
    caption="Fixed post-hoc sample: 500 prompts, eight rollouts each; per-group metrics on 185 mixed groups. Length uses one coefficient calibrated to the eligible-rollout mean learned penalty. Global metrics include prompt weighting; per-group angle is invariant to positive normalization. No retraining."
    (tables/"primary_comparison.md").write_text("\n".join(table)+"\n\n"+caption+"\n",encoding="utf-8")
    with (output/"group_metrics.csv").open(encoding="utf-8",newline="") as f: group=list(csv.DictReader(f))
    with (output/"length_association.csv").open(encoding="utf-8",newline="") as f: ranks=list(csv.DictReader(f))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
    colors={"learned":"#23634f","constant":"#66717c","length_global":"#ba672e","length_group":"#765d9a"}
    def export(fig,name):
        for suffix in ("pdf","svg","png"):fig.savefig(figures/(name+"."+suffix),bbox_inches="tight",dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout="constrained")
    for rule in RULES:
        values=sorted(max(0,float(x["energy"]))*100 for x in group if x["rule"]==rule and x["normalization"]=="std" and x["energy"]!="")
        ax.step(values,np.arange(1,len(values)+1)/len(values)*100,where="post",label=LABELS[rule],color=colors[rule],linestyle="--" if rule=="length_group" else "-",linewidth=1.8)
    ax.set(xlabel="Within-group orthogonal energy (%)",ylabel="Mixed groups at or below x (%)",ylim=(0,102),title="How additional credit varies across the 185 mixed groups")
    ax.grid(alpha=.15);ax.legend(fontsize=8,loc="lower right")
    export(fig,"group_novelty_ecdf")
    fig,axes=plt.subplots(1,2,figsize=(8,4.3),layout="constrained")
    short=["Learned\ndistance","Constant\npenalty","Token\nlength"]
    for ax,norm in zip(axes,NORMS):
        rr=[rules[r]["normalizations"][norm]["global"] for r in s["rules"]]
        within=[100*x["within_energy"] for x in rr];weight=[100*x["weight_energy"] for x in rr]
        ax.bar(short,within,color="#23634f",label="Within-group residual")
        ax.bar(short,weight,bottom=within,color="#c99861",label="Prompt reweighting")
        ax.set(title="Standardized" if norm=="std" else "Mean-only centering",ylabel="Global orthogonal energy (%)")
        ax.grid(axis="y",alpha=.15)
    axes[1].legend(fontsize=8,loc="upper right")
    export(fig,"global_energy_accounting")
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout="constrained")
    for ax,min_wrong,title in zip(axes,(2,3),("At least two incorrect rollouts","At least three incorrect rollouts")):
        values=[float(x["spearman"]) for x in ranks if x["spearman"]!="" and int(x["n_incorrect"])>=min_wrong]
        ax.hist(values,bins=np.linspace(-1,1,11),color="#23634f",edgecolor="white")
        ax.set(xlabel="Within-prompt Spearman correlation",ylabel="Number of defined groups",xlim=(-1.05,1.05),title=f"{title}\n(n = {len(values)})")
        ax.axvline(0,color="#555",linewidth=.7);ax.grid(axis="y",alpha=.15)
    export(fig,"length_association")
    primary=assoc["primary"];three=assoc["at_least_three_incorrect"];learn=rules["learned"]["normalizations"]["std"]["global"]
    line=lambda z: f"median {fmt(z['correlation']['median'])}, IQR [{fmt(z['correlation']['q1'])}, {fmt(z['correlation']['q3'])}], {z['correlation']['defined']} defined groups"
    report=["# Completed CPU credit-assignment audit", "", "This is a descriptive, fixed-sample counterfactual analysis. It does not estimate training performance or identify the cause of the historical task result.","",
      "## Execution and cohort","",f"The sealed protocol was evaluated on {structure['total_groups']} prompts and {structure['total_rollouts']} saved rollouts. There are {structure['mixed_group_denominator']} mixed groups and {structure['eligible_rollouts']} eligible incorrect rollouts. All input identities and historical hashes matched. Independent saved-output validation and a deterministic repeat are recorded in `validation.json`.","",
      "## Primary comparison","",*table,"",caption,"",
      f"The length coefficient is {c['coefficient']:.12g} penalty units per stored generated token. Its numerator is {c['numerator_total_learned_penalty']:.12g}; its denominator is {c['denominator_total_eligible_tokens']} eligible tokens. This is calibration on the same audit sample, not a validated training setting.","",
      "Global energy and relative change are alternative geometric descriptions, not independent replications. The learned standardized row has global energy "+percent(learn["energy"])+"%, relative change "+percent(learn["relative_change"])+"%, and norm ratio "+fmt(learn["norm_ratio"],9)+". Full precision is retained in JSON.","",
      "## Within groups versus across prompts","","Every primary rule has the same per-group angle under both normalization conventions. Mean-only centering retains scale changes. The constant control has zero within-group residual up to floating-point precision, but its prompt-dependent scaling can contribute to global orthogonal energy. `full_geometry.csv` and the energy-accounting figure separate these components.","",
      "## Structural opportunity","",f"Exactly {structure['exactly_one_incorrect_count']}/{structure['mixed_group_denominator']} mixed groups ({percent(structure['exactly_one_incorrect_fraction'])}%) contain one incorrect rollout. Incorrect-only penalties have no orthogonal within-group component there. This does not rule out changes of magnitude or prompt weighting.","", "Incorrect-count histogram: `"+json.dumps(structure["mixed_group_incorrect_histogram"],sort_keys=True)+"`.","",
      "## Association with token length","", "Among incorrect solutions within prompts: "+line(primary)+". Exclusions: `"+json.dumps(primary["exclusions"],sort_keys=True)+"`.","",f"There are {primary['defined_exactly_two_incorrect']} defined groups with exactly two incorrect rollouts; their correlations can only be +1 or -1. The prespecified at-least-three sensitivity gives "+line(three)+".","","Correlation measures association of two orderings. It does not show that the latent representation is only length, nor that either ordering improves learning.","",
      "## Token counting and censoring","","Lengths count all stored completion tokens, including any stored special/stop tokens. An archived local tokenizer verifies 3,952 exact decodes and 48 saved-text prefixes whose remaining decoded suffix is a stop string. There are 3,882 stored terminal end-of-text tokens. The exact historical exporter and tokenizer revision are not reconstructed by that check; see the source manifest.","",
      f"Nominal-cap-reaching samples: {cap['all_rollouts']['nominal_cap_reached']}/{cap['all_rollouts']['denominator']}. Eligible incorrect cap-reaching samples: {cap['eligible_incorrect_rollouts']['nominal_cap_reached']}/{cap['eligible_incorrect_rollouts']['denominator']}. Effective caps and stopping causes remain unknown. These observations do not establish absence of truncation.","",
      f"The wrong-answer length tie fraction is {cap['length_tied_pairs']}/{cap['possible_incorrect_pairs']} = {percent(cap['tied_pair_fraction'])}%. Whole-group cap-exclusion sensitivity status: **{sen['cap_exclusion']['status']}**. "+sen['cap_exclusion'].get('reason','See sensitivity_results.json.'),"",
      "## Prespecified sensitivities","","The per-group matched-length construction is fully reported in `sensitivity_results.json` and the complete geometry table. It is a learned-score-dependent diagnostic construction and does not replace the globally calibrated primary row. The cap sensitivity uses the original global coefficient if triggered; unknown termination is not treated as a confirmed non-length stop.","",
      "## Limits and paper use","","This package adds a second score and a controlled removal of residual variation to the audit. It does not remedy the original single-seed design, the head-gradient lifecycle defect, the evaluation/re-decode discrepancy, the Oracle-pool comparator confound, or missing model-extraction provenance. All-wrong groups remain inactive by construction for every matched rule. No p-values, new confidence intervals, reward tuning, generation, training or policy-gradient measurements were added.","",
      "Use `tables/primary_comparison.tex` and `tables/experiment_results.tex` for paper integration. The original blog and historical source artifacts were not changed. The analysis is complete; a full submission manuscript still requires drafting and review."]
    (output/"analysis_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    tex=[r"% Results excerpt generated from the sealed CPU audit; not a complete paper.",r"\subsection{Offline counterfactual comparison}",
      f"We evaluate three incorrect-only, mixed-group-gated penalties on 500 fixed prompts with eight recorded rollouts each. The length coefficient ({c['coefficient']:.8g}) matches the total learned penalty over the 650 eligible incorrect rollouts. This is an in-sample scale calibration, not a training experiment.",
      r"\input{primary_comparison.tex}",
      "Per-group cosine is invariant to the two positive normalization scalars. Global cosine need not be invariant because prompt weights change; we separately account for within-group residual energy and prompt reweighting.",
      f"Exactly {structure['exactly_one_incorrect_count']} of 185 mixed groups ({percent(structure['exactly_one_incorrect_fraction'])}\\%) contain one incorrect rollout, leaving no within-incorrect variation for these penalties.",
      f"Within-prompt distance--length Spearman correlation has median {fmt(primary['correlation']['median'])} over {primary['correlation']['defined']} defined groups. Restricting to at least three incorrect rollouts gives median {fmt(three['correlation']['median'])} over {three['correlation']['defined']} groups. Neither association establishes useful credit assignment.",
      "All termination reasons and per-request effective caps are unavailable; no stored sequence reaches the configured nominal generation cap. We do not infer that all samples are untruncated. The historical optimization defect and decoder discrepancy preclude attributing the task result to normalization alone."]
    (tables/"experiment_results.tex").write_text("\n\n".join(tex)+"\n",encoding="utf-8")
    claims=[]
    for name,file,key,cohort in [("primary_comparison","summary.json","rules","500 fixed prompt groups; per-group summaries on 185 mixed groups"),("one_incorrect","summary.json","structure.exactly_one_incorrect_count","185 mixed groups"),("length_association","summary.json","length_association","incorrect rollouts within eligible prompts; defined counts reported"),("calibration","calibration.json","coefficient","650 eligible incorrect rollouts"),("censoring","censoring_report.json","all_rollouts","4000 saved completion-token records"),("sensitivity","sensitivity_results.json","length_group","same fixed cohort")]:
        claims.append({"claim":name,"file":file,"sha256":sha(output/file),"key":key,"cohort":cohort,"script":"src/audit.py","script_sha256":sha(run/"src/audit.py"),"status":"completed_cpu_analysis_of_saved_records"})
    write(output/"claim_map.json",{"schema_version":1,"seal_sha256":sha(run/"seal.json"),"claims":claims})
    print("Rendered tables, three vector figure families, report, results excerpt and claim map.")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();render(a.run,a.output)
