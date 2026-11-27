"""CPU-only tests of the actual replay helpers using tiny real Qwen/LoRA modules."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import torch
from replay import weights,output_directions,load,write,require,score,gram_metrics,accumulate,bank_gram

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    torch.manual_seed(29);torch.set_num_threads(2);checks={}
    def close(name,x,y,atol=1e-9,rtol=1e-7):
        xx=np.asarray(x);yy=np.asarray(y)
        require(np.allclose(xx,yy,atol=atol,rtol=rtol),name)
        checks[name]=float(np.max(np.abs(xx-yy)))
    plan=load(Path(__file__).parent/"data/weight_plan.json");basis,coef=weights(plan);names,t=output_directions()
    # Independent h_i fixture, all 1480 rows, to exercise every shuffled weight.
    h=np.random.default_rng(9).normal(size=(1480,23))
    bank=np.zeros((38,23))
    for i in range(1480):bank+=np.outer(coef[i]/1480,h[i])
    full=t@bank
    direct=[]
    for group in plan["groups"]:
        r=np.array(group["correctness"]);a0=np.array(group["binary"])
        b=np.array(group["shaped"]["learned"]);l=np.array(group["shaped"]["length"])
        w=np.array(group["residual"]["learned"]);v=np.array(group["residual"]["length"])
        perm=np.array(group["permutations"])
        direct.extend(np.column_stack([a0,b,group["shaped"]["constant"],l]+[b[p] for p in perm]+[w,v]+[w[p] for p in perm]))
    direct=np.array(direct)
    close("all_70_directions_match_direct_weights",full,direct.T@h/1480)
    close("gram_transformation",t@(bank@bank.T)@t.T,full@full.T)
    metrics=gram_metrics(full@full.T,names)
    for j in range(32):
        aa=full[1];bb=full[4+j]
        close("shuffle_gradient_metric_"+str(j),metrics[f"shuffle_{j:02d}"]["vs_learned"]["relative_difference"],np.linalg.norm(bb-aa)/np.linalg.norm(aa))
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.)
    config._attn_implementation="eager"
    base=Qwen2ForCausalLM(config)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=["q_proj","v_proj"],lora_dropout=0.,task_type="CAUSAL_LM"))
    model=model.double();model.eval();params=[p for p in model.parameters() if p.requires_grad]
    prompts=[[1,2],[3,4,5],[6,7],[8,9]];completions=[[3,4,5],[7,8],[2,3],[4,6,7]]
    gs=[]
    for pp,cc in zip(prompts,completions,strict=True):
        model.zero_grad(set_to_none=True);v=score(model,pp+cc,cc,"cpu");v.backward()
        gs.append(torch.cat([p.grad.flatten() for p in params]).detach().numpy())
        with torch.no_grad():
            logits=model(input_ids=torch.tensor([pp+cc]),use_cache=False).logits[0]
            manual=torch.stack([logits[len(pp)+j-1].float().log_softmax(-1)[target] for j,target in enumerate(cc)]).mean()
        close("causal_mask_"+str(len(gs)),v.item(),manual.item(),atol=1e-6)
    w=np.array([1.,-2.,.5,3.])
    model.zero_grad(set_to_none=True)
    total=sum(float(wi)*score(model,pp+cc,cc,"cpu") for wi,pp,cc in zip(w,prompts,completions,strict=True))
    total.backward();g=torch.cat([p.grad.flatten() for p in params]).detach().numpy()
    close("actual_qwen_lora_weighted_backward",g,w@np.array(gs),atol=1e-7,rtol=2e-5)
    # Check the exact noncontiguous bank slices and checkpoint roundtrip used on GPU.
    bankt=torch.zeros((38,17),dtype=torch.float32);v=torch.randn(38);g=torch.randn(5)
    bankt[:,3:8].addr_(v,g);close("noncontiguous_addr",bankt[:,3:8].numpy(),torch.outer(v,g).numpy(),atol=1e-6)
    cpu_bank=torch.zeros((38,23),dtype=torch.float32)
    sliced=torch.zeros_like(cpu_bank)
    for i in range(1480):
        weight=torch.tensor(coef[i]/1480,dtype=torch.float32)
        gradient=torch.tensor(h[i],dtype=torch.float32)
        accumulate(cpu_bank,weight,gradient)
        for left,right in [(0,7),(7,19),(19,23)]:
            sliced[:,left:right].addr_(weight,gradient[left:right])
    close('cpu_whole_gradient_vs_original_slices',cpu_bank.numpy(),sliced.numpy(),atol=1e-7,rtol=1e-5)
    close('cpu_accumulation_vs_float64_reference',cpu_bank.numpy(),bank,atol=1e-7,rtol=1e-5)
    close('cpu_chunked_gram',bank_gram(cpu_bank,chunk=7),(cpu_bank.double()@cpu_bank.double().T).numpy(),atol=1e-12)
    from safetensors.torch import save_file,load_file
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"state.safetensors";save_file({"bank":bankt},str(path));close("state_roundtrip",bankt.numpy(),load_file(str(path))["bank"].numpy())
        model.save_pretrained(tmp);reloaded=PeftModel.from_pretrained(Qwen2ForCausalLM(config),tmp,is_trainable=True)
        require(any(p.requires_grad for p in reloaded.parameters()),"Adapter load path")
    write(a.output,dict(status="PASS",scope="CPU tiny randomly initialized Qwen/LoRA software validation; no 7B scientific gradient result",checks=checks))
    print("CPU_REPLAY_TESTS_PASS",len(checks))

if __name__=="__main__":main()
