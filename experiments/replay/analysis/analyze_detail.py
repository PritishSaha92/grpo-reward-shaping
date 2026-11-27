"""Post-result reporting closure and independent numerical audit. CPU only."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PILOT=HERE/'pilot_reference'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data): (HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(v):
    finite=[float(x) for x in v if x is not None]
    return dict(defined=len(finite),undefined=len(v)-len(finite),**dict(zip(
        ['min','q1','median','q3','max'],map(float,np.quantile(finite,[0,.25,.5,.75,1])) if finite else [None]*5)))

result=load(ROOT/'run/result.json')
stats=load(ROOT/'run/group_statistics.json')
records=load(ROOT/'code/data/replay_records.json')
plan=load(ROOT/'code/data/weight_plan.json')
summary=load(ROOT/'results/summary.json')
assert result['status']=='COMPLETE' and len(stats)==len(records)==len(plan['groups'])==185
K=np.asarray(result['direction_gram']); B=np.asarray(result['basis_gram'])
names=result['direction_names']; index={n:i for i,n in enumerate(names)}
bnames=result['basis_names']; bi={n:i for i,n in enumerate(bnames)}
# Independent Euclidean embedding of the stored Gram checks pair arithmetic.
ev,U=np.linalg.eigh((K+K.T)/2)
V=U*np.sqrt(np.maximum(ev,0))[None,:]
max_pair_error=0.
def pair(ref,comp):
    global max_pair_error
    i,j=index[ref],index[comp];a=K[i,i];b=K[j,j];ab=K[i,j]
    valid=a>0 and b>0
    c=float(ab/math.sqrt(a*b)) if valid else None
    rho=math.sqrt(b/a) if a>0 else None
    d=math.sqrt(max(a+b-2*ab,0)/a) if a>0 else None
    if valid:
        independent=[np.dot(V[i],V[j])/np.linalg.norm(V[i])/np.linalg.norm(V[j]),
            np.linalg.norm(V[j])/np.linalg.norm(V[i]),np.linalg.norm(V[j]-V[i])/np.linalg.norm(V[i])]
        for x,y in zip([c,rho,d],independent):
            assert math.isclose(x,y,abs_tol=2e-10,rel_tol=2e-9),(ref,comp,x,y)
            max_pair_error=max(max_pair_error,abs(x-y))
    return dict(reference=ref,compared=comp,cosine=c,norm_ratio=rho,relative_difference=d,
                reference_norm=math.sqrt(a),compared_norm=math.sqrt(b))
allpairs=[pair(ref,n) for ref in ['binary','learned','residual_learned'] for n in names]
csvout('all_reference_comparisons.csv',allpairs)
distribution=[]
for kind in ['full','residual']:
    ds=[f'shuffle_{j:02d}' if kind=='full' else f'residual_shuffle_{j:02d}' for j in range(32)]
    for ref in ['binary','learned','residual_learned']:
        pp=[pair(ref,n) for n in ds]
        for metric in ['cosine','norm_ratio','relative_difference','compared_norm']:
            distribution.append(dict(direction_type=kind,reference=ref,metric=metric,**quantile([p[metric] for p in pp])))
csvout('permutation_quantiles.csv',distribution)
same_difference_errors=[];empirical_squared_differences=[]
for j in range(32):
    full_delta=V[index[f'shuffle_{j:02d}']]-V[index['learned']]
    residual_delta=V[index[f'residual_shuffle_{j:02d}']]-V[index['residual_learned']]
    squared=float(full_delta@full_delta);res_squared=float(residual_delta@residual_delta)
    assert math.isclose(squared,res_squared,rel_tol=1e-10,abs_tol=1e-16)
    same_difference_errors.append(abs(squared-res_squared));empirical_squared_differences.append(squared)
assert math.isclose(float(np.mean(empirical_squared_differences)),summary['finite_32_draw_comparison']['empirical_mean_squared_difference_from_learned'],rel_tol=1e-10,abs_tol=1e-16)
assert math.isclose(summary['finite_32_draw_comparison']['exact_expected_squared_difference_from_learned'],
    summary['exact_reference']['learned']['exact_expected_squared_norm']+K[index['residual_learned'],index['residual_learned']],rel_tol=1e-12)

T=np.zeros((70,38));T[0,0]=1;T[1,[1,4]]=1;T[2,3]=1;T[3,[2,5]]=1
for j in range(32):T[4+j,[1,6+j]]=1;T[38+j,6+j]=1
T[36,4]=1;T[37,5]=1
np.testing.assert_allclose(K,T@B@T.T,rtol=1e-12,atol=1e-16)
decomposition=[]
for rule in ['learned','length']:
    l=bi['aligned_'+rule];w=bi['residual_'+rule]
    aligned_difference=B[l,l]+B[0,0]-2*B[0,l]
    cross=2*(B[l,w]-B[0,w])
    residual=B[w,w];actual=K[index[rule],index[rule]]+K[0,0]-2*K[0,index[rule]]
    assert math.isclose(actual,aligned_difference+residual+cross,abs_tol=1e-17,rel_tol=1e-10)
    decomposition.append(dict(rule=rule,
        aligned_vs_binary_cosine=float(B[0,l]/math.sqrt(B[0,0]*B[l,l])),
        aligned_norm_over_binary=math.sqrt(B[l,l]/B[0,0]),
        aligned_difference_over_binary=math.sqrt(max(aligned_difference,0)/B[0,0]),
        delta_squared=actual,aligned_delta_squared=aligned_difference,residual_squared=residual,cross_term=cross,
        caution='Cross term can be negative; these are not nonnegative attribution shares.'))
csvout('gradient_decomposition.csv',decomposition)

group_rows=[];perm_energy_error=0.;perm_dot_error=0.;expected={r:[] for r in ['learned','length']}
all_scores=[];all_norms=[];token_rows=[]
for st,rec,g in zip(stats,records,plan['groups'],strict=True):
    assert st['group_id']==rec['group_id']==g['group_id']
    R=np.array(g['correctness']); wrong=R==0;m=int(wrong.sum());A=np.array(g['binary'])
    np.testing.assert_allclose(A,(R-R.mean())/(R.std()+1e-4),atol=1e-14,rtol=1e-14)
    h2=np.array(st['completion_gradient_squared_norms']);all_norms.extend(np.sqrt(h2));all_scores.extend(st['completion_scores'])
    scatter=float(h2[wrong].sum()-st['scatter']['squared_norm_gradient_sum']/m)
    assert math.isclose(scatter,st['scatter']['centered_scatter'],abs_tol=5e-15,rel_tol=1e-10)
    row=dict(group_id=g['group_id'],n_wrong=m,n_correct=8-m,scatter=st['scatter']['centered_scatter'])
    for rule in ['learned','length']:
        W=np.array(g['residual'][rule]);shaped=np.array(g['shaped'][rule]);alpha=g['alpha'][rule]
        np.testing.assert_allclose(shaped,alpha*A+W,atol=2e-15,rtol=2e-15)
        assert abs(W.sum())<1e-13 and abs(W@A)<1e-13
        assert np.max(np.abs(W[~wrong]))<1e-13
        w2=float(W[wrong]@W[wrong]);e=0. if m==1 else w2*row['scatter']/(m-1)/1480**2
        assert math.isclose(e,st['reference'][rule]['expected_squared_gradient_norm'],rel_tol=1e-12,abs_tol=1e-20)
        row[rule+'_weight_energy']=w2;row[rule+'_reference']=e;expected[rule].append(e)
    learned=np.array(g['shaped']['learned']);wl=np.array(g['residual']['learned'])
    for perm in g['permutations']:
        assert sorted(perm)==list(range(8)) and np.array_equal(R[perm],R)
        perm_energy_error=max(perm_energy_error,abs(float(learned[perm]@learned[perm]-learned@learned)))
        perm_dot_error=max(perm_dot_error,abs(float(A@learned[perm]-A@learned)))
        np.testing.assert_allclose(learned[perm],g['alpha']['learned']*A+wl[perm],atol=2e-15,rtol=2e-15)
    group_rows.append(row)
    if m>1 and row['length_weight_energy']==0:
        assert len({len(x['token_ids']) for x in rec['samples'] if x['correctness']==0})==1
    for i,sample in enumerate(rec['samples']):
        L=len(sample['token_ids']);trim=sample['dropped_prompt_tokens'];P=len(rec['prompt_token_ids'])
        assert trim==max(0,P+L-4096) and P-trim>0
        assert sample['correctness']==g['correctness'][i] and sample['sample_idx']==i
        token_rows.append(dict(group_id=rec['group_id'],sample_idx=i,correctness=sample['correctness'],
            completion_tokens=L,prompt_tokens=P,prompt_tokens_removed=trim,total_tokens=P-trim+L,
            mean_log_probability=st['completion_scores'][i],gradient_squared_norm=h2[i]))
csvout('per_group_audit.csv',group_rows);csvout('per_completion_audit.csv',token_rows)
strata=[]
for m in range(1,8):
    rows=[r for r in group_rows if r['n_wrong']==m]
    q=dict(n_wrong=m,groups=len(rows),incorrect_completions=m*len(rows),
           scatter_sum=math.fsum(r['scatter'] for r in rows))
    for rule in expected:
        q[rule+'_nonzero_groups']=sum(r[rule+'_weight_energy']>0 for r in rows)
        q[rule+'_reference']=math.fsum(r[rule+'_reference'] for r in rows)
        q[rule+'_reference_share']=q[rule+'_reference']/math.fsum(expected[rule])
    strata.append(q)
csvout('incorrect_count_breakdown.csv',strata)
concentration={}
for rule,es in expected.items():
    total=math.fsum(es);np.testing.assert_allclose(total,summary['exact_reference'][rule]['exact_expected_squared_norm'],rtol=1e-12)
    ranking=sorted(group_rows,key=lambda x:(-x[rule+'_reference'],x['group_id']))
    concentration[rule]=dict(zero_reference_groups=sum(e==0 for e in es),
        top_1_share=math.fsum(r[rule+'_reference'] for r in ranking[:1])/total,
        top_5_share=math.fsum(r[rule+'_reference'] for r in ranking[:5])/total,
        top_10_share=math.fsum(r[rule+'_reference'] for r in ranking[:10])/total,
        top_10=[dict(group_id=r['group_id'],n_wrong=r['n_wrong'],reference=r[rule+'_reference']) for r in ranking[:10]])

for name,digest in load(PILOT/'SHA256_MANIFEST.json').items(): assert sha(PILOT/name)==digest
old_result=load(PILOT/'received/primary/runs/primary/result.json')
old_records=load(PILOT/'code/data/replay_records.json');old_plan=load(PILOT/'code/data/weight_plan.json')
recmap={r['group_id']:r for r in records};gmap={g['group_id']:g for g in plan['groups']}
assert all(x==recmap[x['group_id']] for x in old_records)
assert all(x==gmap[x['group_id']] for x in old_plan['groups'])
pilot_runtime=old_result['identity']['runtime'];runtime=result['identity']['runtime']
pilot_checks=dict(shared_record_groups=len(old_records),shared_completion_records_identical=True,
    shared_weights_and_permutations_identical=True,
    native_runtime_identical=pilot_runtime==runtime,
    model_revisions_identical=old_result['identity']['preflight']['model_revisions']==result['identity']['preflight']['model_revisions'],
    parameter_layout_identical=load(PILOT/'received/primary/runs/primary/parameter_layout.json')==load(ROOT/'run/parameter_layout.json'),
    prior_divisor=old_result['identity']['common_divisor'],current_divisor=result['identity']['common_divisor'],
    subgroup_gradient_reproduction='Not available: prior pilot did not store per-completion scores/norms or subgroup cross-Grams; new complete-cohort bank does not retain the old-subset aggregate separately.')
assert pilot_checks['native_runtime_identical'] and pilot_checks['model_revisions_identical'] and pilot_checks['parameter_layout_identical']
def metrics(k,ns):
    ix={x:i for i,x in enumerate(ns)}
    def p(a,b):
        aa=k[ix[a],ix[a]];bb=k[ix[b],ix[b]];ab=k[ix[a],ix[b]]
        return dict(cos=float(ab/np.sqrt(aa*bb)),rho=float(np.sqrt(bb/aa)),D=float(np.sqrt(max(aa+bb-2*ab,0)/aa)))
    residual=k[ix['residual_learned'],ix['residual_learned']]
    sq=[k[ix[f'residual_shuffle_{j:02d}'],ix[f'residual_shuffle_{j:02d}']] for j in range(32)]
    ds=[p('learned',f'shuffle_{j:02d}')['D'] for j in range(32)]
    return dict(learned_binary_cosine=p('binary','learned')['cos'],learned_binary_relative_difference=p('binary','learned')['D'],
        learned_residual_norm_over_binary=p('binary','residual_learned')['rho'],
        learned_residual_binary_cosine=p('binary','residual_learned')['cos'],
        learned_length_residual_cosine=p('residual_learned','residual_length')['cos'],
        length_learned_relative_difference=p('learned','length')['D'],
        median_shuffled_learned_relative_difference=float(np.median(ds)),
        shuffles_larger_residual=int(sum(x>residual for x in sq)),
        shuffles_farther_than_length=int(sum(x>p('learned','length')['D'] for x in ds)))
pilotmetrics=metrics(np.array(old_result['direction_gram']),old_result['direction_names']);fullmetrics=metrics(K,names)
csvout('pilot_vs_full.csv',[dict(metric=k,pilot_32=pilotmetrics[k],full_185=fullmetrics[k]) for k in pilotmetrics])

# Transport logs were verified in the original extraction audit and are not
# redistributed. This public check validates scientific records only.
cast=load(ROOT/'weight_precision_validation.json')
trimmed=[t for t in token_rows if t['prompt_tokens_removed']]
report=dict(status='PASS',input_result_sha256=sha(ROOT/'run/result.json'),
    comparisons=210,quantile_rows=len(distribution),undefined_comparisons=sum(p['cosine'] is None for p in allpairs),
    max_independent_pair_error=max_pair_error,max_advantage_permutation_norm_squared_error=perm_energy_error,
    max_same_difference_squared_error=max(same_difference_errors),
    max_advantage_permutation_binary_dot_error=perm_dot_error,
    group_count=185,completion_count=1480,log_events_ordered_and_complete="not rechecked; transport logs excluded",
    tokens=dict(saved_completion_tokens=sum(t['completion_tokens'] for t in token_rows),
        trimmed_records=len(trimmed),trimmed_groups=len(set(t['group_id'] for t in trimmed)),
        total_prompt_tokens_removed=sum(t['prompt_tokens_removed'] for t in trimmed),
        max_prompt_tokens_removed=max(t['prompt_tokens_removed'] for t in token_rows),
        completion_length=quantile([t['completion_tokens'] for t in token_rows]),
        maximum_context=max(t['total_tokens'] for t in token_rows)),
    completion_mean_log_probability=quantile(all_scores),completion_gradient_norm=quantile(all_norms),
    concentration=concentration,pilot_checks=pilot_checks,
    additional_zero_length_groups=[g['group_id'] for g in group_rows if g['n_wrong']>1 and g['length_reference']==0],
    coefficient_cast_max_relative_l2_error=cast['max_relative_l2_cast_error'],
    precision_limit='Coefficient conversion only; no higher-precision full accumulation rerun and no bound on all BF16/FP32 numerical error.',
    scientific_limit='No policy updates, independent validation outcomes, utility ranking or significance test; exact reference is conditional on fixed computed gradients.')
write('detailed_checks.json',report)
write('concentration.json',concentration)
write('pilot_checks.json',pilot_checks)
write('gradient_decomposition.json',decomposition)
print(json.dumps(report,indent=2))
