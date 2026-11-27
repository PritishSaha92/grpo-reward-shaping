"""Fixed-sample LoRA gradient replay. No optimizer, generation, training or paid-job launch."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import numpy as np

# Required by deterministic CUDA GEMM; set before importing torch/CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE=Path(__file__).resolve().parent
VERSIONS={"torch":"2.9.1","transformers":"4.57.6","peft":"0.19.1","huggingface-hub":"0.36.0",
          "tokenizers":"0.22.2","accelerate":"1.12.0","safetensors":"0.7.0","numpy":"1.26.4"}

def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def jsonsha(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def weights(plan):
    """38 basis directions, versus 70 redundant full/residual accumulators."""
    names=["binary","aligned_learned","aligned_length","constant","residual_learned","residual_length"]+[f"residual_shuffle_{j:02d}" for j in range(32)]
    data=[]
    for group in plan["groups"]:
        a=np.array(group["binary"],float);r=np.array(group["correctness"])
        wl=np.array(group["residual"]["learned"]);wt=np.array(group["residual"]["length"])
        columns=[a,group["alpha"]["learned"]*a,group["alpha"]["length"]*a,
                 np.array(group["shaped"]["constant"]),wl,wt]
        for j,perm in enumerate(group["permutations"]):
            pp=np.array(perm);require(np.array_equal(r[pp],r),"Permutation crosses outcome classes")
            columns.append(wl[pp])
        block=np.column_stack(columns)
        require(np.all(np.isfinite(block)),"Nonfinite weight")
        for key,ix,jx in (("learned",1,4),("length",2,5)):
            require(np.allclose(block[:,ix]+block[:,jx],group["shaped"][key],atol=1e-10,rtol=1e-8),"Weight decomposition")
        data.extend(block)
    return names,np.array(data,dtype=np.float64)


def output_directions():
    names=["binary","learned","constant","length"]+[f"shuffle_{j:02d}" for j in range(32)]+["residual_learned","residual_length"]+[f"residual_shuffle_{j:02d}" for j in range(32)]
    t=np.zeros((len(names),38));t[0,0]=1;t[1,[1,4]]=1;t[2,3]=1;t[3,[2,5]]=1
    for j in range(32):t[4+j,[1,6+j]]=1
    t[36,4]=1;t[37,5]=1
    for j in range(32):t[38+j,6+j]=1
    return names,t


def preflight(data,models,require_base=False):
    for rel,expected in load(data/"seal.json")["files"].items():require(sha(data/rel)==expected,"Data hash mismatch: "+rel)
    for rel,record in load(data/"download_manifest.json").items():require(sha(models/rel)==record["sha256"],"Model file mismatch: "+rel)
    pins=load(data/"model_pins.json");cfg=load(models/"adapter/adapter_config.json")
    require(cfg["base_model_name_or_path"]==pins["base"]["repo_id"],"Adapter base mismatch")
    require(cfg["r"]==64 and cfg["lora_dropout"]==0 and cfg["bias"]=="none","Adapter configuration changed")
    require(not cfg.get("use_dora") and not cfg.get("use_rslora"),"Unexpected LoRA variant")
    from safetensors import safe_open
    with safe_open(models/"adapter/adapter_model.safetensors",framework="np") as f:
        keys=list(f.keys());n=sum(math.prod(f.get_slice(k).get_shape()) for k in keys)
        require(all("lora_A" in k or "lora_B" in k for k in keys),"Unexpected adapter tensors")
    require(n==161480704,"Unexpected trainable adapter size")
    index=load(models/"base/model.safetensors.index.json")
    missing=[]
    for name in sorted(set(index["weight_map"].values())):
        path=models/"base"/name;remote=pins["base"]["remote_files"][name]
        if path.is_file():require(sha(path)==remote["sha256"],"Base shard hash mismatch: "+name)
        else:missing.append(name)
    if require_base:require(not missing,"Base model shards missing; hydrate them on CPU before starting GPU time")
    records=load(data/"replay_records.json");plan=load(data/"weight_plan.json");names,coef=weights(plan)
    require(len(records)==len(plan["groups"])==185 and coef.shape==(1480,38),"Sample dimensions")
    seen=set()
    for rec,group in zip(records,plan["groups"],strict=True):
        require(rec["group_id"]==group["group_id"] and rec["group_id"] not in seen,"Group identity")
        seen.add(rec["group_id"]);require(len(rec["samples"])==8,"Sample count")
        pp=rec["prompt_token_ids"]
        for i,s in enumerate(rec["samples"]):
            require(s["sample_idx"]==i and s["correctness"]==group["correctness"][i],"Sample mismatch")
            ids=s["token_ids"];drop=max(0,len(pp)+len(ids)-4096)
            require(drop==s["dropped_prompt_tokens"] and len(pp)>drop and len(ids)>0,"Token/context mismatch")
    return dict(status="PASS" if not missing else "CPU_READY_BASE_DOWNLOAD_PENDING",missing_base_shards=missing,
        adapter_parameters=n,n_rollouts=1480,basis_directions=38,accumulator_GiB=n*38*4/2**30,
        data_seal_sha256=sha(data/"seal.json"),source_sha256=sha(__file__),weight_sha256=jsonsha(coef.tolist()),
        model_revisions={k:v["revision"] for k,v in pins.items()})


def hydrate(data,models):
    from huggingface_hub import hf_hub_download
    pins=load(data/"model_pins.json");manifest=load(data/"download_manifest.json")
    require(pins["adapter"]["repo_id"] != "REPLAY_ADAPTER", "Adapter source is withheld in this review export. Stage the exact hashed files locally; see docs/MODEL_REPLAY.md.")
    for rel,record in manifest.items():
        kind,name=rel.split("/",1);path=models/rel
        if path.is_file() and sha(path)==record["sha256"]:continue
        hf_hub_download(pins[kind]["repo_id"],name,revision=pins[kind]["revision"],local_dir=models/kind)
        require(sha(path)==record["sha256"],"Hydrated file mismatch")
    for name in sorted(set(load(models/"base/model.safetensors.index.json")["weight_map"].values())):
        path=models/"base"/name;expected=pins["base"]["remote_files"][name]["sha256"]
        if not path.is_file() or sha(path)!=expected:
            hf_hub_download(pins["base"]["repo_id"],name,revision=pins["base"]["revision"],local_dir=models/"base")
        require(sha(path)==expected,"Hydrated base weight mismatch")
        print("Verified base shard",name,flush=True)


def native_snapshot():
    return dict(kind="isolated-native-venv",python=sys.version,executable_sha256=sha(sys.executable),
        platform=platform.platform(),packages={d.metadata["Name"].lower():d.version
        for d in importlib.metadata.distributions() if d.metadata["Name"]})


def runtime(container_digest,native_manifest=None):
    import torch
    require(bool(container_digest)!=bool(native_manifest),"Choose actual container digest OR native runtime manifest")
    if native_manifest:
        snapshot=native_snapshot()
        require(load(native_manifest)==snapshot,"Native runtime manifest differs from installed environment")
        provenance=dict(native_manifest_sha256=sha(native_manifest),native_snapshot=snapshot)
    else:
        require(re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}",container_digest) is not None,"Supply actual built container digest")
        provenance=dict(container_digest_operator_supplied=container_digest)
    versions={p:importlib.metadata.version(p) for p in VERSIONS}
    for p,v in VERSIONS.items():require(versions[p].split("+")[0]==v,"Version mismatch: "+p)
    require(torch.cuda.is_available(),"CUDA GPU required")
    return dict(**provenance,versions=versions,python=platform.python_version(),
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        gpu_memory=torch.cuda.get_device_properties(0).total_memory,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))


def accumulate(bank,weight,gradient):
    """Same float32 outer-product update, with one transfer per completion."""
    import torch
    with torch.no_grad():
        bank.addr_(weight.to(device=bank.device,dtype=torch.float32),
                   gradient.to(device=bank.device,dtype=torch.float32))


def bank_gram(bank,chunk=262144):
    import torch
    gram=torch.zeros((bank.shape[0],bank.shape[0]),dtype=torch.float64,device=bank.device)
    with torch.no_grad():
        for left in range(0,bank.shape[1],chunk):
            block=bank[:,left:left+chunk].double()
            gram.add_(block@block.T)
    return gram.cpu().numpy()


def gram_metrics(gram,names):
    def pair(i,j):
        aa,bb=float(gram[i,i]),float(gram[j,j]);ab=float(gram[i,j])
        require(aa>=-1e-12 and bb>=-1e-12,"Invalid Gram diagonal")
        aa=max(0.,aa);bb=max(0.,bb)
        if aa==0 or bb==0:return dict(cosine=None,norm_ratio=None,relative_difference=None,norm_a=math.sqrt(aa),norm_b=math.sqrt(bb))
        return dict(cosine=float(np.clip(ab/math.sqrt(aa*bb),-1,1)),norm_ratio=math.sqrt(bb/aa),
            relative_difference=math.sqrt(max(0.,aa+bb-2*ab)/aa),norm_a=math.sqrt(aa),norm_b=math.sqrt(bb))
    out={}
    for j,name in enumerate(names):out[name]=dict(vs_binary=pair(0,j))
    for j in range(32):
        out[f"shuffle_{j:02d}"]["vs_learned"]=pair(1,4+j)
        out[f"residual_shuffle_{j:02d}"]["vs_original_residual"]=pair(36,38+j)
    out["length"]["vs_learned"]=pair(1,3)
    out["residual_length"]["vs_original_residual"]=pair(36,37)
    return out


def score(model,full_ids,completion_ids,device):
    import torch
    import torch.nn.functional as F
    ids=torch.tensor([full_ids],dtype=torch.long,device=device)
    targets=torch.tensor(completion_ids,dtype=torch.long,device=device)
    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    # Causal logit at position t predicts token at t+1. Include every saved
    # completion token, including any saved stop suffix, with equal token weight.
    logits=out.logits[0,len(full_ids)-len(completion_ids)-1:-1,:]
    require(logits.shape[0]==len(completion_ids),"Causal target alignment")
    return -F.cross_entropy(logits.float(),targets,reduction="mean")

