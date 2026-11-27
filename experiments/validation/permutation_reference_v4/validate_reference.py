"""Independent saved-Gram summaries and exhaustive finite permutation checks."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT = ROOT.parent / 'grpo-mechanism-audit/analysis/math4ai_2027/assignment_extension_v1/thor_replay/received/primary/runs/primary/result.json'

def compute(path):
    r=json.loads(path.read_text())
    assert r['status']=='COMPLETE' and r['groups_complete']==32
    names=r['direction_names']; gram=np.asarray(r['direction_gram'],dtype=np.float64)
    idx={name:i for i,name in enumerate(names)}
    def dot(a,b):return float(gram[idx[a],idx[b]])
    def norm(a):return float(np.sqrt(dot(a,a)))
    def cosine(a,b):return dot(a,b)/(norm(a)*norm(b))
    def diff(a,b):return np.sqrt(max(0.,dot(a,a)+dot(b,b)-2*dot(a,b)))
    bn=norm('binary'); ln=norm('learned'); sn=norm('residual_learned')
    residual_names=[f'residual_shuffle_{i:02d}' for i in range(32)]
    shuffle_names=[f'shuffle_{i:02d}' for i in range(32)]
    raw_error=max(abs(diff(a,'learned')-diff(b,'residual_learned')) for a,b in zip(shuffle_names,residual_names))
    assert raw_error<1e-12
    scatter=[{'name':n,'norm_over_binary':norm(n)/bn,'cosine_with_learned_residual':cosine(n,'residual_learned')} for n in ['residual_learned','residual_length']+residual_names]
    values={
        'residual_norm_over_binary_norm':sn/bn,
        'residual_binary_cosine':cosine('residual_learned','binary'),
        'difference_denominator_factor':ln/sn,
        'max_same_difference_norm_error':float(raw_error),
        'learned_residual_norm_smaller_than_n_shuffles':sum(norm(n)>sn for n in residual_names),
        'learned_residual_squared_norm_over_mean_shuffle_squared_norm':sn**2/np.mean([norm(n)**2 for n in residual_names]),
        'mean_shuffle_residual_cosine_with_learned':float(np.mean([cosine(n,'residual_learned') for n in residual_names])),
        'length_D_smaller_than_n_shuffles':int(sum(diff(n,'learned')>diff('length','learned') for n in shuffle_names)),
        'length_D_vs_learned':float(diff('length','learned')/ln),
    }
    for key,ref in [('residual_norm_over_binary_norm',0.20049771912307798),('residual_binary_cosine',0.017516146170482125),('learned_residual_squared_norm_over_mean_shuffle_squared_norm',0.7648344569914564)]:
        np.testing.assert_allclose(values[key],ref,rtol=1e-11,atol=1e-12)
    assert values['learned_residual_norm_smaller_than_n_shuffles']==31
    assert values['length_D_smaller_than_n_shuffles']==28

    rng=np.random.default_rng(20261127)
    def expected(w,h,N=1):
        if len(w)<2:return 0.
        return float((w@w)/(len(w)-1)*np.sum((h-h.mean(0))**2)/N**2)
    def enumerate_s(w,h,N=1):
        return np.array([w[list(p)]@h/N for p in itertools.permutations(range(len(w)))])
    errors=[];trials=0;perms=0
    for m in range(1,8):
        for trial in range(3):
            w=rng.normal(size=m);w-=w.mean()
            h=rng.normal(size=(m,4))
            N=11
            ss=enumerate_s(w,h,N);observed=w@h/N
            true=expected(w,h,N)
            np.testing.assert_allclose(ss.mean(0),0,atol=1e-12)
            second=float(np.mean(np.sum(ss**2,axis=1)))
            errors.append(abs(second-true))
            np.testing.assert_allclose(second,true,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.mean(np.sum((ss-observed)**2,axis=1)),true+observed@observed,rtol=1e-12,atol=1e-12)
            # Direct centered scatter and raw norms agree.
            np.testing.assert_allclose(np.sum((h-h.mean(0))**2),np.sum(h*h)-np.sum(h.sum(0)**2)/m,rtol=1e-12,atol=1e-12)
            # Identical gradients annihilate zero-sum weights.
            np.testing.assert_allclose(enumerate_s(w,np.ones_like(h),N),0,atol=1e-12)
            trials+=1;perms+=len(ss)
    wa=np.array([2.,-1.,-1.]);ha=np.array([[1.,2.],[3.,-1.],[-2.,0.]])
    wb=np.array([1.,-1.]);hb=np.array([[3.,1.],[-1.,2.]])
    sa=enumerate_s(wa,ha,7);sb=enumerate_s(wb,hb,7)
    combined=np.array([a+b for a,b in itertools.product(sa,sb)])
    np.testing.assert_allclose(np.mean(np.sum(combined**2,axis=1)),expected(wa,ha,7)+expected(wb,hb,7),rtol=1e-12)
    counter=enumerate_s(wa,wa[:,None]).ravel()
    assert sorted(counter.tolist())==[-3.,-3.,-3.,-3.,6.,6.]
    assert counter.mean()==0
    np.testing.assert_allclose(np.sign(counter).mean(),-1/3)
    return {
        'status':'PASS',
        'scope':'Post-hoc presentation amendment; saved 32-group results, not new GPU evidence or significance tests.',
        'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'summaries':values,'scatter_points':scatter,
        'synthetic_validation':{'fixtures':trials,'enumerated_permutations':perms,'max_second_moment_absolute_error':max(errors),'independent_groups_and_batch_factor':'PASS','identical_gradients_and_single_incorrect':'PASS'},
        'expected_cosine_counterexample':{'values':counter.tolist(),'expected_vector':0.,'expected_cosine':float(np.sign(counter).mean())},
        'exact_model_reference_available':False,
    }

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=HERE/'report.json')
    args=parser.parse_args()
    report=compute(args.result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'summaries':report['summaries'],'synthetic_validation':report['synthetic_validation']},indent=2))
