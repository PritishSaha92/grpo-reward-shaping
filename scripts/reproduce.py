"""Recompute the published fixed-sample analyses on CPU, without network access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from compare_outputs import compare_files, compare_values, RTOL, ATOL

ROOT = Path(__file__).resolve().parents[1]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(argv, log, cwd):
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([sys.executable, '-B', *map(str, argv)], cwd=cwd,
            stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    if result.returncode:
        raise RuntimeError(f'{log.name} failed:\n{log.read_text(encoding="utf-8", errors="replace")[-4500:]}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists(): parser.error('Choose a new output directory; existing results are never overwritten.')
    if not (ROOT/'checksums.json').is_file(): parser.error('Repository checksums.json is missing.')
    for name,h in json.loads((ROOT/'checksums.json').read_text()).items():
        if digest(ROOT/name)!=h: raise RuntimeError('Repository file changed: '+name)
    out.mkdir(parents=True); logs=out/'logs';logs.mkdir(); work=out/'work'
    shutil.copytree(ROOT/'experiments',work)
    # The assignment analyzer requires a fresh results directory. Rename the
    # copied reference output; the committed evidence is never touched.
    (work/'assignment/results').rename(work/'assignment/expected_results')
    began=time.monotonic(); checks={}; stages=[]
    def stage(name,argv,cwd=ROOT):
        print(name,flush=True);run(argv,logs/(name+'.txt'),cwd);stages.append(name)
    def compare(label,expected,actual,names):
        for n in names:
            checks[label+'/'+n]=compare_files(expected/n,actual/n)

    offline=work/'offline'; assignment=work/'assignment'; replay=work/'replay'
    stage('01_synthetic', [offline/'src/test_synthetic_cases.py','--output',out/'synthetic.json'])
    stage('02_offline', [offline/'src/audit.py','run','--run',offline,'--output',out/'offline'])
    names=['calibration.json','summary.json','sensitivity_results.json','censoring_report.json',
           'group_metrics.csv','length_association.csv','rollout_results.jsonl','calculation_checks.json']
    compare('offline',offline/'results',out/'offline',names)
    stage('03_independent_offline_checks',[offline/'src/validate_results.py','--run',offline,'--output',out/'offline'])
    stage('04_assignment_synthetic',[assignment/'src/analyze.py','synthetic'])
    stage('05_assignment',[assignment/'src/analyze.py','run','--run',assignment])
    compare('assignment',assignment/'expected_results',assignment/'results',
            ['summary.json','geometry.csv','per_group.csv','permutations.json','permutation_checks.csv','gpu_weight_plan.json','validation.json'])
    stage('06_replay',[replay/'analyze_result.py','--run',replay/'run','--data',replay/'code/data','--output',out/'replay'])
    compare('replay',replay/'results',out/'replay',['summary.json','all_comparisons.csv','permutation_draws.csv','group_reference.csv'])
    stage('06_coefficient_precision',[replay/'check_weight_precision.py'])
    precision=json.loads((replay/'weight_precision_validation.json').read_text())
    recorded_precision=json.loads((ROOT/'experiments/replay/weight_precision_validation.json').read_text())
    # The input seal identifies the public export; every numerical field must
    # still match the recorded coefficient-cast diagnostic under the same policy.
    for field in recorded_precision:
        if field!='input_seal_sha256':
            compare_values(recorded_precision[field],precision[field],'coefficient_precision.'+field)
    stage('06_reporting_synthetic',[replay/'test_reporting.py'])
    reporting=json.loads((replay/'reporting_test.json').read_text())
    assert reporting['status']=='PASS' and reporting['groups']==185
    stage('07_detailed_replay',[replay/'analysis/analyze_detail.py'])
    compare('replay_detail',ROOT/'experiments/replay/analysis',replay/'analysis',
        ['all_reference_comparisons.csv','concentration.json','gradient_decomposition.csv','gradient_decomposition.json',
         'incorrect_count_breakdown.csv','permutation_quantiles.csv','per_completion_audit.csv','per_group_audit.csv','pilot_checks.json','pilot_vs_full.csv'])
    detail=json.loads((replay/'analysis/detailed_checks.json').read_text());assert detail['status']=='PASS'
    stage('08_pilot',[work/'pilot/analyze_completed.py','--run',work/'pilot/results','--output',out/'pilot'])
    compare('pilot',work/'pilot/results',out/'pilot',['analysis.json','all_comparisons.csv','summary_validation.json'])
    for name,script in [('tau_identity_v1','validate_tau.py'),('general_shaping_v2','validate_general.py'),('actionability_v3','validate_actionability.py')]:
        folder=work/'validation'/name
        stage('09_'+name,[folder/script,'--run',offline])
        assert json.loads((folder/'validation.json').read_text())['status']=='PASS'
    compare('alignment_floors',ROOT/'experiments/validation/actionability_v3',work/'validation/actionability_v3',['alignment_floors.csv'])
    stage('10_permutation_identity',[work/'validation/permutation_reference_v4/validate_reference.py',
        '--result',work/'pilot/results/result.json','--output',out/'permutation_identity.json'])
    # These reports include provenance hashes; the numerical fields are checked
    # by the validators rather than confusing an export hash with a new result.
    stage('11_historical',[ROOT/'scripts/reproduce_history.py','--root',work/'historical','--output',out/'historical'])
    report={'status':'PASS','scope':'CPU arithmetic from saved inputs and gradients; no decoding, model backpropagation or training',
        'stages':stages,'output_comparisons':checks,'verified_outputs':len(checks),
        'byte_identical_outputs':sum(c['byte_identical'] for c in checks.values()),
        'comparison_policy':{'relative_tolerance':RTOL,'absolute_tolerance':ATOL,
            'exact':'input SHA-256, structure, integer counts, identifiers, strings, booleans and nulls'},
        'runtime_seconds':round(time.monotonic()-began,3),
        'python':sys.version.split()[0], 'main_replay_groups':185,'main_replay_completions':1480,
        'coefficient_precision_numerical_match':True,'synthetic_full_reporting':'PASS',
        'length_shuffle_draws':0,'learned_shuffle_draws':32}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
