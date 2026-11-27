from pathlib import Path
import json, csv, hashlib, argparse
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'offline'
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=RUN)
RUN=parser.parse_args().run.resolve()
groups=defaultdict(list)
for line in (RUN/'joined_rollouts.jsonl').read_text().splitlines():
    r=json.loads(line); groups[r['group_id']].append(r)
saved={(r['group_id'],r['rule'],r['normalization']):r for r in csv.DictReader((RUN/'results/group_metrics.csv').open())}
summary=list(csv.DictReader((RUN/'results/tables/full_geometry.csv').open()))
lam=sum(r['learned_penalty'] for g in groups.values() for r in g)/sum(r['stored_generated_token_count'] for g in groups.values() for r in g if r['group_type']=='mixed' and not r['correctness'])
errors=[]; checks=0
def check(a,b):
    global checks
    checks+=1; errors.append(float(abs(a-b)))
    assert np.isclose(a,b,atol=1e-10,rtol=1e-8),(a,b)
def cos(a,b):
    return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
pool=defaultdict(list)
for gid,rows in sorted(groups.items()):
    rows=sorted(rows,key=lambda r:r['sample_idx'])
    R=np.array([r['correctness'] for r in rows],float); x=R-R.mean()
    if R.mean() in [0,1]:
        assert all(saved[gid,rule,n]['undefined_reason']=='both_zero' for rule in ['learned','constant','length_global','length_group'] for n in ['std','mean'])
        continue
    wrong=R==0
    q=np.array([r['learned_penalty'] for r in rows])
    length=np.array([r['stored_generated_token_count'] for r in rows],float)*wrong
    rules={'learned':q,'constant':wrong*q[wrong].mean(),'length_global':lam*length,'length_group':length*q[wrong].sum()/length.sum()}
    for rule,q in rules.items():
        mu=q[wrong].mean(); sd=q[wrong].std(); pi=R.mean()
        tau=sd/(1+mu); predicted=pi/(pi+tau*tau)
        b=R-q; b-=b.mean()
        check(cos(x,b)**2,predicted)
        check(np.dot(b,b),len(R)*(1-pi)*((1+mu)**2*pi+sd**2))
        for n in ['std','mean']:
            A=x/(R.std()+1e-4) if n=='std' else x
            B=b/((R-q).std()+1e-4) if n=='std' else b
            check(cos(A,B),float(saved[gid,rule,n]['cosine']))
            check(cos(A,B)**2,predicted)
            pool[rule,n].append((A,B))
for row in summary:
    pairs=pool[row['rule'],row['normalization']]
    A=np.concatenate([a for a,b in pairs]); B=np.concatenate([b for a,b in pairs])
    c=cos(A,B); rho=np.linalg.norm(B)/np.linalg.norm(A); alpha=np.dot(A,B)/np.dot(A,A)
    within=weight=0
    for a,b in pairs:
        ap=np.dot(a,b)/np.dot(a,a)
        within+=np.linalg.norm(b-ap*a)**2
        weight+=(ap-alpha)**2*np.dot(a,a)
    within/=np.dot(B,B); weight/=np.dot(B,B)
    for key,value in [('cosine',c),('norm_ratio',rho),('energy',1-c*c),('within_energy',within),('weight_energy',weight)]:
        check(value,float(row[key]))
    check(within+weight,1-c*c)
    check(np.linalg.norm(B-A)/np.linalg.norm(A),np.sqrt(max(0,1+rho*rho-2*rho*c)))
# Synthetic checks; these do not select a coefficient for the recorded data.
R=np.array([1.,1.,0.,0.]); x=R-R.mean()
prev=1.
for beta in [0.,.1,1.,2.,5.]:
    q=np.array([0,0,beta,3*beta]); b=R-q; b-=b.mean(); c=cos(x,b)
    check(c*c,.5/(.5+(beta/(1+2*beta))**2))
    assert c<=prev+1e-10 and c>=np.sqrt(2/3)-1e-10
    prev=c
prev=0.
for shift in [0.,1.,2.,5.]:
    q=np.array([0,0,1+shift,3+shift]); b=R-q; b-=b.mean(); c=cos(x,b)
    check(c*c,.5/(.5+(1/(3+shift))**2)); assert c>=prev; prev=c
check(cos(x,np.array([1.5,1.5,-.5,-2.5])),3/np.sqrt(11))
for R in [np.array([1.,1.,0.,0.]),np.array([1.,1.,1.,0.])]:
    for m in [0.,2.]:
        x=R-R.mean(); b=R-(1-R)*m; b-=b.mean()
        check(cos(x,b),1.)
out=Path(__file__).parent
inputs=[RUN/'joined_rollouts.jsonl',RUN/'results/group_metrics.csv',RUN/'results/tables/full_geometry.csv',Path(__file__),out/'AMENDMENT.md']
hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p.relative_to(RUN)) if p.is_relative_to(RUN) else p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
report={'status':'PASS','mixed_groups':185,'rules':4,'normalizations':2,'checks':checks,'max_absolute_error':max(errors),'tolerance':{'atol':1e-10,'rtol':1e-8},'hashes':hashes,'scope':'Post-result identity validation; no new empirical outcomes'}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'SHA256SUMS.txt').write_text('\n'.join(f'{v}  {k}' for k,v in hashes.items())+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2))
