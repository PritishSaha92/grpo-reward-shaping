"""Exercise the full reporting route on synthetic gradients, without any GPU."""
from pathlib import Path
import tempfile,subprocess,sys,json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from core import weights,output_directions,load,write,sha

def main():
    plan=load(HERE/'code/data/weight_plan.json');_,coef=weights(plan);names,T=output_directions()
    h=np.random.default_rng(876).normal(size=(1480,19));bank=coef.T@h/1480;K=(T@bank)@(T@bank).T
    stats=[];exact={k:0. for k in ['learned','length']}
    for j,g in enumerate(plan['groups']):
        block=h[j*8:j*8+8];mask=np.array(g['correctness'])==0;hw=block[mask];m=len(hw)
        scatter=dict(n_wrong=m,centered_scatter=float(np.square(hw-hw.mean(0)).sum()),
            sum_individual_squared_norms=float(np.square(hw).sum()),squared_norm_gradient_sum=float(np.square(hw.sum(0)).sum()))
        refs={}
        for rule in exact:
            w=np.array(g['residual'][rule])[mask];e=0. if m==1 else float(w@w)/(m-1)*scatter['centered_scatter']/1480**2
            refs[rule]=dict(expected_squared_gradient_norm=e);exact[rule]+=e
        stats.append(dict(group_id=g['group_id'],source_index=j,scatter=scatter,reference=refs,
            completion_scores=[-1.]*8,completion_gradient_squared_norms=np.square(block).sum(1).tolist()))
    with tempfile.TemporaryDirectory() as td:
        run=Path(td)/'run';run.mkdir();out=Path(td)/'report';write(run/'group_statistics.json',stats)
        write(run/'result.json',dict(status='COMPLETE',groups_complete=185,identity=dict(group_indices=list(range(185)),
            common_divisor=1480,canary=False,preflight=dict(data_seal_sha256=sha(HERE/'code/data/seal.json'))),
            group_statistics_sha256=sha(run/'group_statistics.json'),basis_gram=(bank@bank.T).tolist(),
            direction_gram=K.tolist(),direction_names=names,exact_expected_squared_residual_norm=exact))
        cp=subprocess.run([sys.executable,str(HERE/'analyze_result.py'),'--run',str(run),'--data',str(HERE/'code/data'),
                           '--output',str(out)],capture_output=True,text=True)
        assert cp.returncode==0,cp.stdout+cp.stderr
        result=load(out/'summary.json');assert result['groups']==185 and result['single_incorrect']==53
        for rule in exact:
            np.testing.assert_allclose(result['exact_reference'][rule]['exact_expected_squared_norm'],exact[rule],rtol=1e-12)
        assert result['permutation_distributions']['full_relative_difference']['defined']==32
        write(HERE/'reporting_test.json',dict(status='PASS',scope='Synthetic 1480-gradient full reporting route; not a model result',
            groups=185,single_incorrect=53,directions=70,permutations=32))
    print('FULL_REPORTING_TEST_PASS')
if __name__=='__main__':main()
