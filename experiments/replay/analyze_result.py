"""Independent saved-result validation and reporting; never runs the model."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    r=load(a.run/'result.json');stats=load(a.run/'group_statistics.json');plan=load(a.data/'weight_plan.json')
    assert r['status'] in ['COMPLETE','CANARY_COMPLETE']
    assert sha(a.run/'group_statistics.json')==r['group_statistics_sha256']
    ident=r['identity'];indices=ident['group_indices'];N=ident['common_divisor']
    assert len(stats)==r['groups_complete']==len(indices) and N==1480
    assert indices==(load(a.data/'provenance.json')['canary_group_indices'] if ident['canary'] else list(range(185)))
    for name,h in load(a.data/'seal.json')['files'].items():assert sha(a.data/name)==h,name
    assert ident['preflight']['data_seal_sha256'] in {sha(a.data/'seal.json'), load(a.data/'seal.json').get('original_seal_sha256')}
    B=np.array(r['basis_gram'],float);K=np.array(r['direction_gram'],float);names=r['direction_names'];idx={n:i for i,n in enumerate(names)}
    assert B.shape==(38,38) and K.shape==(70,70) and np.isfinite(K).all()
    np.testing.assert_allclose(B,B.T,atol=1e-12,rtol=1e-10)
    eig=np.linalg.eigvalsh(B);assert eig.min()>=-1e-9*max(float(np.trace(B)),1e-30)
    T=np.zeros((70,38));T[0,0]=1;T[1,[1,4]]=1;T[2,3]=1;T[3,[2,5]]=1
    for j in range(32):T[4+j,[1,6+j]]=1;T[38+j,6+j]=1
    T[36,4]=1;T[37,5]=1
    np.testing.assert_allclose(K,T@B@T.T,atol=1e-12,rtol=1e-10)
    contributions=[];exact={k:[] for k in ['learned','length']}
    for st,j in zip(stats,indices,strict=True):
        g=plan['groups'][j];mask=np.array(g['correctness'])==0;m=int(mask.sum());s=st['scatter']
        assert st['source_index']==j and st['group_id']==g['group_id'] and m==s['n_wrong']
        assert len(st['completion_scores'])==len(st['completion_gradient_squared_norms'])==8
        assert np.isfinite(st['completion_scores']).all() and np.all(np.array(st['completion_gradient_squared_norms'])>0)
        sumsq=math.fsum(np.array(st['completion_gradient_squared_norms'])[mask]);M2=sumsq-s['squared_norm_gradient_sum']/m
        assert abs(sumsq-s['sum_individual_squared_norms'])<=1e-8*max(sumsq,1e-20)+1e-15
        assert abs(M2-s['centered_scatter'])<=1e-8*max(sumsq,1e-20)+1e-15
        row=dict(group_id=g['group_id'],n_wrong=m,centered_scatter=s['centered_scatter'])
        for rule in exact:
            w=np.array(g['residual'][rule]);ss=float(w[mask]@w[mask]);e=0. if m==1 else ss/(m-1)*s['centered_scatter']/N**2
            np.testing.assert_allclose(e,st['reference'][rule]['expected_squared_gradient_norm'],rtol=1e-10,atol=1e-18)
            exact[rule].append(e);row[rule+'_exact_expected_squared_norm']=e
        contributions.append(row)
    exact={k:math.fsum(v) for k,v in exact.items()}
    for k,v in exact.items():np.testing.assert_allclose(v,r['exact_expected_squared_residual_norm'][k],rtol=1e-10,atol=1e-18)
    def pair(ref,comp):
        i,j=idx[ref],idx[comp];aa=max(float(K[i,i]),0.);bb=max(float(K[j,j]),0.);ab=float(K[i,j])
        return dict(comparison=comp+' / '+ref,cosine=float(np.clip(ab/math.sqrt(aa*bb),-1,1)) if aa*bb else None,
            norm_ratio=math.sqrt(bb/aa) if aa else None,relative_difference=math.sqrt(max(0.,aa+bb-2*ab)/aa) if aa else None,
            reference_norm=math.sqrt(aa),compared_norm=math.sqrt(bb))
    comparisons=[pair('binary',n) for n in names]
    comparisons+=[pair('learned','length'),pair('residual_learned','residual_length')]
    shuffles=[]
    for j in range(32):
        full=pair('learned',f'shuffle_{j:02d}');res=pair('residual_learned',f'residual_shuffle_{j:02d}')
        comparisons.extend([full,res]);assert full['relative_difference'] is not None
        if res['relative_difference'] is not None:
            np.testing.assert_allclose(full['relative_difference']*full['reference_norm'],res['relative_difference']*res['reference_norm'],atol=1e-11,rtol=1e-8)
        shuffles.append(dict(draw=j,full_cosine=full['cosine'],full_relative_difference=full['relative_difference'],
            residual_cosine=res['cosine'],residual_norm=res['compared_norm'],residual_norm_ratio=res['norm_ratio'],
            residual_squared_norm=res['compared_norm']**2))
    binary_norm=math.sqrt(K[0,0]);reference={}
    for rule in exact:
        i=idx['residual_'+rule];obs=float(K[i,i]);e=exact[rule]
        reference[rule]=dict(observed_squared_norm=obs,exact_expected_squared_norm=e,
            observed_over_expected_squared_norm=obs/e if e else None,root_expected_squared_norm=math.sqrt(e),
            residual_norm_over_binary_norm=math.sqrt(obs)/binary_norm,
            residual_binary_cosine=pair('binary','residual_'+rule)['cosine'])
    learned_sq=float(K[idx['residual_learned'],idx['residual_learned']]);lengthD=pair('learned','length')['relative_difference']
    shuffled_sq=[x['residual_squared_norm'] for x in shuffles]
    denominator=float(K[idx['learned'],idx['learned']])
    distributions={}
    for name in ['full_cosine','full_relative_difference','residual_cosine','residual_norm','residual_norm_ratio','residual_squared_norm']:
        valid=[x[name] for x in shuffles if x[name] is not None]
        distributions[name]=dict(defined=len(valid),undefined=32-len(valid),**dict(zip(
            ['min','q1','median','q3','max'],map(float,np.quantile(valid,[0,.25,.5,.75,1],method='linear')) if valid else [None]*5)))
    summary=dict(status='PASS',result_status=r['status'],groups=len(indices),completions=8*len(indices),common_divisor=N,
        result_sha256=sha(a.run/'result.json'),exact_reference=reference,
        primary={rule:pair('binary',rule) for rule in ['learned','constant','length']},
        permutation_distributions=distributions,
        residual_cross_rule=pair('residual_learned','residual_length'),length_vs_learned=pair('learned','length'),
        finite_32_draw_comparison=dict(mean_squared_residual_norm=float(np.mean(shuffled_sq)),
            mean_over_exact_expectation=float(np.mean(shuffled_sq))/exact['learned'] if exact['learned'] else None,
            exact_expected_squared_difference_from_learned=exact['learned']+learned_sq,
            empirical_mean_squared_difference_from_learned=float(np.mean([x['full_relative_difference']**2*denominator for x in shuffles])),
            learned_residual_smaller_than_draws=sum(x>learned_sq for x in shuffled_sq),
            length_closer_to_learned_than_draws=sum(x['full_relative_difference']>lengthD for x in shuffles)),
        gram_min_eigenvalue=float(eig.min()),single_incorrect=sum(x['n_wrong']==1 for x in contributions),
        scope='Conditional random-assignment second moment and descriptive fixed-sample gradients; no usefulness or significance claim')
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'summary.json',summary)
    for name,rows in [('all_comparisons.csv',comparisons),('permutation_draws.csv',shuffles),('group_reference.csv',contributions)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
