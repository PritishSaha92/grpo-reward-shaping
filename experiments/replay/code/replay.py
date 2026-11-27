"""All mixed-group replay with exact conditional permutation second moments."""
from core import *

def authorize_gpu(rt):
    # PyTorch returns the UUID without NVIDIA's display prefix "GPU-".
    expected=os.environ.get('GRPO_REPLAY_GPU_UUID', '').removeprefix('GPU-')
    require(bool(expected), 'Set GRPO_REPLAY_GPU_UUID to the device explicitly authorized for this new execution')
    require(rt['gpu_uuid'].removeprefix('GPU-')==expected,'Unauthorized GPU')
    require(rt['visible_devices']=='GPU-'+expected,'Unexpected visible GPU selection')

def vector_sq(x,chunk=262144):
    import torch
    total=0.
    for left in range(0,x.numel(),chunk):
        v=x[left:left+chunk].double();total+=float(torch.dot(v,v))
    return total

class WrongScatter:
    """Stable float64 Welford scatter, plus the subtraction-identity check."""
    def __init__(self,nparam,chunk=262144):
        import torch
        self.mean=torch.zeros(nparam,dtype=torch.float64,device='cpu')
        self.n=0;self.m2=0.;self.sum_sq=0.;self.chunk=chunk
    def add(self,g):
        import torch
        require(g.device.type=='cpu','Scatter input must be on CPU')
        self.n+=1;delta_sq=0.;sq=0.
        for left in range(0,g.numel(),self.chunk):
            x=g[left:left+self.chunk].double();mu=self.mean[left:left+self.chunk]
            delta=x-mu;delta_sq+=float(torch.dot(delta,delta));sq+=float(torch.dot(x,x))
            mu.add_(delta,alpha=1./self.n)
        self.m2+=(self.n-1)/self.n*delta_sq;self.sum_sq+=sq
    def finish(self):
        require(self.n>=1,'No incorrect gradients')
        mean_sq=vector_sq(self.mean,self.chunk)
        subtraction=self.sum_sq-self.n*mean_sq
        error=abs(subtraction-self.m2)
        require(error<=1e-8*max(self.sum_sq,1e-20)+1e-15,'Scatter identity mismatch')
        require(math.isfinite(self.m2) and self.m2>=0,'Invalid scatter')
        return dict(n_wrong=self.n,centered_scatter=self.m2,sum_individual_squared_norms=self.sum_sq,
                    squared_norm_gradient_sum=self.n**2*mean_sq,subtraction_scatter=subtraction,
                    identity_absolute_error=error)

def reference_terms(group,scatter,divisor):
    r=np.array(group['correctness']);mask=r==0;m=int(mask.sum())
    require(m==scatter['n_wrong'],'Wrong-gradient count differs')
    out={}
    for rule in ['learned','length']:
        w=np.array(group['residual'][rule],dtype=np.float64)
        require(np.all(np.abs(w[~mask])<1e-12),'Correct-class residual not zero')
        require(abs(float(w[mask].sum()))<1e-10,'Residual not centered')
        ss=float(w[mask]@w[mask]);q=0. if m==1 else ss/(m-1)*scatter['centered_scatter']/divisor**2
        require(m>1 or ss==0.,'Single-incorrect residual must be zero')
        out[rule]=dict(residual_squared_norm=ss,residual_sum=float(w[mask].sum()),
                       expected_squared_gradient_norm=q)
    return out

def verify_source():
    manifest=HERE/'source_manifest.json'
    require(manifest.is_file(),'Missing frozen source manifest')
    for name,h in load(manifest).items():require(sha(HERE/name)==h,'Source hash mismatch: '+name)
    return sha(manifest)

def save_checkpoint(out,bank,statistics,groups_complete,fingerprint,slot):
    from safetensors.torch import save_file
    bankname=f'state_{slot}.safetensors';statsname=f'state_{slot}.groups.json'
    temp=out/(bankname+'.tmp');save_file({'bank':bank.detach().cpu().contiguous()},str(temp))
    os.replace(temp,out/bankname)
    write(out/(statsname+'.tmp'),statistics);os.replace(out/(statsname+'.tmp'),out/statsname)
    state=dict(groups_complete=groups_complete,fingerprint=fingerprint,state_file=bankname,
               state_sha256=sha(out/bankname),statistics_file=statsname,statistics_sha256=sha(out/statsname))
    write(out/'progress.tmp.json',state);os.replace(out/'progress.tmp.json',out/'progress.json')
    return state

def restore_checkpoint(out,fingerprint,nparam):
    from safetensors.torch import load_file
    state=load(out/'progress.json');require(state['fingerprint']==fingerprint,'Resume fingerprint')
    require(sha(out/state['state_file'])==state['state_sha256'],'Resume bank hash')
    require(sha(out/state['statistics_file'])==state['statistics_sha256'],'Resume statistics hash')
    bank=load_file(str(out/state['state_file']),device='cpu')['bank'];stats=load(out/state['statistics_file'])
    require(tuple(bank.shape)==(38,nparam),'Resume bank shape')
    require(len(stats)==state['groups_complete'],'Resume statistics length')
    return bank,stats,state['groups_complete']

def load_policy(models):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    base=AutoModelForCausalLM.from_pretrained(str(models/'base'),local_files_only=True,trust_remote_code=False,
         torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    model=PeftModel.from_pretrained(base,str(models/'adapter'),is_trainable=True,local_files_only=True)
    model.config.use_cache=False;require(getattr(model.config,'attention_dropout',0.)==0.,'Attention dropout')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    for mod in model.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0.
    model.train()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    require(all('lora_A' in n or 'lora_B' in n for n,p in params),'Unexpected trainable tensor')
    require(sum(p.numel() for n,p in params)==161480704,'LoRA parameter count')
    with safe_open(models/'adapter/adapter_model.safetensors',framework='pt',device='cpu') as stored:
        seen=set()
        for name,p in params:
            p.data=p.data.float();key=name.replace('.lora_A.default.','.lora_A.').replace('.lora_B.default.','.lora_B.')
            require(key in stored.keys() and torch.equal(p.detach().cpu(),stored.get_tensor(key).float()),'Adapter tensor identity: '+key)
            seen.add(key)
        require(seen==set(stored.keys()),'Incomplete loaded adapter')
    return model,params

def execute(args):
    import torch
    verified=preflight(args.data,args.models,True);source_hash=verify_source();rt=runtime(None,args.native_runtime)
    authorize_gpu(rt)
    require(rt['gpu_memory']>=40*2**30,'Insufficient GPU capacity')
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(20261126);torch.cuda.manual_seed_all(20261126)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    records=load(args.data/'replay_records.json');plan=load(args.data/'weight_plan.json')
    basis,coef=weights(plan);divisor=len(records)*8
    indices=load(args.data/'provenance.json')['canary_group_indices'] if args.canary else list(range(len(records)))
    identity=dict(preflight=verified,source_manifest_sha256=source_hash,runtime=rt,
        canary=args.canary,group_indices=indices,n_groups=len(indices),common_divisor=divisor,
        estimator='fixed-sample untempered completion-mean LoRA score gradient; no KL/head loss',
        precision='BF16 base; FP32 LoRA/basis bank; FP64 chunked Gram and streaming gradient scatter',
        attention='sdpa',dropout='zero',cpu_threads=args.cpu_threads,checkpoint_every=args.checkpoint_every)
    fp=jsonsha(identity);out=args.output
    if out.exists():
        require(args.resume and load(out/'identity.json')==identity,'Existing output or changed resume identity')
    else:out.mkdir(parents=True);write(out/'identity.json',identity)
    model,params=load_policy(args.models);nparam=sum(p.numel() for _,p in params)
    write(out/'parameter_layout.json',[dict(name=n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype)) for n,p in params])
    if args.resume and (out/'progress.json').is_file():bank,stats,start=restore_checkpoint(out,fp,nparam)
    else:bank=torch.zeros((38,nparam),dtype=torch.float32);stats=[];start=0
    require(start<=len(indices),'Bad resume offset')
    bank_bytes=38*nparam*4
    require(shutil.disk_usage(out).free>3*bank_bytes+10*2**30,'Insufficient atomic-checkpoint headroom')
    print(json.dumps(dict(event='model_ready',groups=len(indices),divisor=divisor,start=start,bank_GiB=bank_bytes/2**30)),flush=True)
    began=time.monotonic()
    for pos in range(start,len(indices)):
        gj=indices[pos];rec=records[gj];group=plan['groups'][gj]
        scatter=WrongScatter(nparam);scores=[];norms=[]
        for i,s in enumerate(rec['samples']):
            prompt=rec['prompt_token_ids'][s['dropped_prompt_tokens']:]
            model.zero_grad(set_to_none=True);v=score(model,prompt+s['token_ids'],s['token_ids'],'cuda')
            require(torch.isfinite(v).item(),'Nonfinite score');v.backward()
            with torch.no_grad():
                require(all(p.grad is not None for _,p in params),'Missing LoRA gradient')
                g=torch.cat([p.grad.detach().reshape(-1) for _,p in params]).float()
                require(torch.isfinite(g).all().item() and torch.count_nonzero(g).item()>0,'Invalid LoRA gradient')
                gc=g.cpu();del g
                accumulate(bank,torch.tensor(coef[gj*8+i]/divisor,dtype=torch.float32),gc)
                n2=vector_sq(gc)
                if s['correctness']==0:scatter.add(gc)
                scores.append(float(v.detach()));norms.append(n2)
            del v,gc
            print(json.dumps(dict(event='rollout_complete',group=pos+1,groups=len(indices),sample=i+1,elapsed_s=time.monotonic()-began)),flush=True)
        st=scatter.finish();refs=reference_terms(group,st,divisor);del scatter
        stats.append(dict(group_id=rec['group_id'],source_index=gj,scatter=st,reference=refs,
                          completion_scores=scores,completion_gradient_squared_norms=norms))
        print(json.dumps(dict(event='group_compute_complete',group=pos+1,groups=len(indices),elapsed_s=time.monotonic()-began)),flush=True)
        if (pos+1)%args.checkpoint_every and pos+1<len(indices):continue
        require(shutil.disk_usage(out).free>bank_bytes+10*2**30,'Checkpoint disk headroom lost')
        slot=((pos+args.checkpoint_every)//args.checkpoint_every)%2
        state=save_checkpoint(out,bank,stats,pos+1,fp,slot)
        print(json.dumps(dict(event='group_checkpointed',**state,elapsed_s=time.monotonic()-began,
                        peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    gram=bank_gram(bank);names,t=output_directions();full=t@gram@t.T
    reference={rule:math.fsum(s['reference'][rule]['expected_squared_gradient_norm'] for s in stats) for rule in ['learned','length']}
    write(out/'group_statistics.json',stats)
    result=dict(status='CANARY_COMPLETE' if args.canary else 'COMPLETE',identity=identity,basis_names=basis,
        basis_gram=gram.tolist(),direction_names=names,direction_gram=full.tolist(),metrics=gram_metrics(full,names),
        groups_complete=len(stats),exact_expected_squared_residual_norm=reference,
        group_statistics_sha256=sha(out/'group_statistics.json'),elapsed_s=time.monotonic()-began,
        peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
        limitations='Fixed computed gradients, reconstructed prompts; conditional random-assignment reference is not learning quality')
    write(out/'result.json',result);print(result['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','run'])
    p.add_argument('--data',type=Path,default=HERE/'data');p.add_argument('--models',type=Path,required=True)
    p.add_argument('--native-runtime',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=8);p.add_argument('--checkpoint-every',type=int,default=8)
    p.add_argument('--canary',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();require(a.checkpoint_every>=1,'Bad checkpoint interval')
    if a.command=='preflight':
        report=preflight(a.data,a.models,True);report['source_manifest_sha256']=verify_source();write(a.output,report);print('PREFLIGHT_PASS')
    else:
        require(a.native_runtime is not None,'Pinned native runtime required');execute(a)
if __name__=='__main__':main()
