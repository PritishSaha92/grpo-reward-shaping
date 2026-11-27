"""Quantify the protocol's FP32 coefficient cast; no sample/model changes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'code'))
from core import load,weights
plan=load(root/'code/data/weight_plan.json');names,w=weights(plan);a=w/1480;b=a.astype(np.float32).astype(np.float64)
rows=[]
for j,name in enumerate(names):
    x=a[:,j];delta=b[:,j]-x;norm=np.linalg.norm(x)
    rows.append(dict(direction=name,max_absolute_coefficient_error=float(abs(delta).max()),
        relative_l2_cast_error=float(np.linalg.norm(delta)/norm) if norm else None,
        max_within_group_coefficient_sum_drift=float(abs(delta.reshape(185,8).sum(1)).max())))
out=dict(status='PASS',input_seal_sha256=hashlib.sha256((root/'code/data/seal.json').read_bytes()).hexdigest(),
    coefficients=1480*38,divisor=1480,directions=rows,
    max_relative_l2_cast_error=max(x['relative_l2_cast_error'] or 0 for x in rows),
    scope='FP64 coefficients divided by 1480 then cast to FP32 as executed. Quantifies coefficient conversion only, not cumulative floating-point accumulation error.')
(root/'weight_precision_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='directions'},indent=2))
