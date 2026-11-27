"""Check the motivating results from saved records, with no model execution."""
from pathlib import Path
import argparse,importlib.util,json,math
import numpy as np

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    audit=module(a.root/'reference_code/analyze_coverage_advantage_novelty.py','historical_audit')
    payload,groups=audit.load_groups(a.root/'inputs/geometry_records.json')
    metrics=audit.compute_core_metrics(groups,0.1)
    expected=json.loads((a.root/'results/coverage_advantage.json').read_text())
    compared=[]
    def walk(actual,reference,path=''):
        if isinstance(actual,dict):
            for k,v in actual.items():
                if k in reference:walk(v,reference[k],path+'/'+k)
        elif isinstance(actual,(float,int)) and not isinstance(actual,bool):
            assert math.isclose(actual,reference,rel_tol=1e-10,abs_tol=1e-10),(path,actual,reference);compared.append(path)
        elif isinstance(actual,list):
            assert len(actual)==len(reference),path
            for i,(x,y) in enumerate(zip(actual,reference)):walk(x,y,path+'/'+str(i))
        else:assert actual==reference,(path,actual,reference)
    walk(metrics,expected)
    # Retain the estimator and fixed RNG used by the saved paired comparison.
    paired=module(a.root/'reference_code/analyze_math500_n8.py','paired_stats')
    recorded_pairs=json.loads((a.root/'results/paired_results.json').read_text())
    runs={name:paired.load_run(a.root/'inputs'/(name+'.json')) for name in recorded_pairs['runs']}
    run_summaries={name:{k:run[k] for k in recorded_pairs['runs'][name]} for name,run in runs.items()}
    walk(run_summaries,recorded_pairs['runs'],'task/runs')
    comparisons={}
    for name in recorded_pairs['comparisons']:
        right,left=name.split('_minus_');lm,rm,la,ra=paired.align_runs(runs[left],runs[right])
        boot=paired.paired_bootstrap(lm,rm,recorded_pairs['bootstrap']['resamples'],recorded_pairs['bootstrap']['seed'])
        binary=dict(delta=runs[right]['pass_at_8']-runs[left]['pass_at_8'],**paired.exact_mcnemar(la,ra))
        comparisons[name]=dict(mean_correctness=boot,pass_at_8=binary)
    walk(comparisons,recorded_pairs['comparisons'],'task/comparisons')
    (a.output/'paired_results.json').write_text(json.dumps(dict(runs=run_summaries,comparisons=comparisons),indent=2)+'\n')
    # Training coverage is independently reconstructed from the supplied CSV.
    import subprocess,sys
    result=subprocess.run([sys.executable,'-B',str(a.root/'reference_code/analyze_g7_training_coverage.py'),
        '--history',str(a.root/'inputs/training_history.csv'),'--output',str(a.output/'training_coverage.json')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    training=json.loads((a.output/'training_coverage.json').read_text()); recorded=json.loads((a.root/'results/training_coverage.json').read_text())
    for k in training:
        if k not in {'provenance','metadata','history','history_sha256','source','source_sha256','input_sha256'} and k in recorded:walk(training[k],recorded[k],'training/'+k)
    (a.output/'geometry.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (a.output/'verification.json').write_text(json.dumps({'status':'PASS','numeric_fields_checked':len(compared),
        'scope':'Recorded geometry/AUROC, training coverage and fixed-seed paired statistics; no model or training rerun'},indent=2)+'\n')
    print('Historical geometry, AUROC, coverage and paired statistics verified:',len(compared),'numeric fields')
if __name__=='__main__':main()
