"""Exhaustive reference identity, streaming statistics and atomic resume tests."""
import argparse,itertools,tempfile
from pathlib import Path
import numpy as np
import torch
from replay import WrongScatter,reference_terms,save_checkpoint,restore_checkpoint,load,write,require

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);rng=np.random.default_rng(726);checks={}
    def close(name,x,y,atol=1e-10,rtol=1e-8):
        xx=np.asarray(x);yy=np.asarray(y);require(np.allclose(xx,yy,atol=atol,rtol=rtol),name)
        checks[name]=float(np.max(np.abs(xx-yy)))
    for m in range(1,8):
        h=rng.normal(size=(m,11));w=rng.normal(size=m);w-=w.mean()
        s=WrongScatter(11,chunk=4)
        for x in h:s.add(torch.tensor(x))
        st=s.finish();close(f'scatter_{m}',st['centered_scatter'],np.square(h-h.mean(0)).sum())
        g=dict(correctness=[1]+[0]*m,residual={r:[0.]+w.tolist() for r in ['learned','length']})
        q=reference_terms(g,st,17)['learned']['expected_squared_gradient_norm']
        vals=np.array([w[list(perm)]@h/17 for perm in itertools.permutations(range(m))])
        close(f'exhaustive_second_moment_{m}',np.square(vals).sum(1).mean(),q)
        close(f'exhaustive_mean_zero_{m}',vals.mean(0),np.zeros(11))
        original=w@h/17
        close(f'exhaustive_difference_{m}',np.square(vals-original).sum(1).mean(),q+original@original)
    # Independent-group cross terms vanish; a nonuniform sample need not do so.
    h1=rng.normal(size=(3,7));h2=rng.normal(size=(2,7));w1=np.array([2.,-1.,-1.]);w2=np.array([1.,-1.])
    v1=np.array([w1[list(p)]@h1 for p in itertools.permutations(range(3))]);v2=np.array([w2[list(p)]@h2 for p in itertools.permutations(range(2))])
    combined=np.array([x+y for x in v1 for y in v2])
    expected=(w1@w1)/2*np.square(h1-h1.mean(0)).sum()+(w2@w2)*np.square(h2-h2.mean(0)).sum()
    close('independent_group_second_moment',np.square(combined).sum(1).mean(),expected)
    # Zero mean vector is insufficient for zero expected cosine.
    scalar=[float(w1[list(p)]@w1) for p in itertools.permutations(range(3))]
    close('expected_cosine_counterexample',np.sign(scalar).mean(),-1/3)
    for scale in [0.,1.,1e8]:
        h=scale+rng.normal(size=(7,15))*.01;s=WrongScatter(15,chunk=4)
        for x in h:s.add(torch.tensor(x))
        st=s.finish()
        close(f'aligned_streaming_scatter_{scale}',st['centered_scatter'],np.square(h-h.mean(0)).sum(),atol=2e-8,rtol=1e-5)
    # A constant score has a zero residual even with nonzero gradient scatter.
    zero=dict(correctness=[1,0,0],residual={r:[0.,0.,0.] for r in ['learned','length']})
    close('constant_reference_zero',reference_terms(zero,dict(n_wrong=2,centered_scatter=9.),1480)['learned']['expected_squared_gradient_norm'],0.)
    plan=load(Path(__file__).parent/'data/weight_plan.json')
    for g in plan['groups']:
        m=g['correctness'].count(0.);mask=np.array(g['correctness'])==0
        for rule in ['learned','length']:
            w=np.array(g['residual'][rule]);require(abs(w[mask].sum())<1e-10,'Real residual mean')
            if m==1:require(np.all(w==0),'Single-incorrect residual')
    checks['real_plan_groups_validated']=len(plan['groups'])
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);b=torch.randn(38,31);stats=[dict(group_id='a',scatter={'n_wrong':2})]
        save_checkpoint(out,b,stats,1,'test-fingerprint',0)
        bb,ss,n=restore_checkpoint(out,'test-fingerprint',31)
        close('checkpoint_bank_roundtrip',bb.numpy(),b.numpy());require(ss==stats and n==1,'Stats roundtrip')
        # Progress remains usable while an inactive slot is partially written.
        (out/'state_1.safetensors.tmp').write_bytes(b'incomplete')
        bb,ss,n=restore_checkpoint(out,'test-fingerprint',31);close('incomplete_other_slot_ignored',bb.numpy(),b.numpy())
        failed=False
        try:restore_checkpoint(out,'different-fingerprint',31)
        except ValueError:failed=True
        require(failed,'Changed identity accepted')
    write(a.output,dict(status='PASS',scope='CPU software and finite-permutation identities only',checks=checks))
    print('REFERENCE_TESTS_PASS',len(checks))
if __name__=='__main__':main()
