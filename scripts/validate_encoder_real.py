"""Full-checkpoint hidden_states[50] parity for all three campaign prompts."""
import gc
import json
import os
import sys
from pathlib import Path
from common import ROOT, OUT, SANA_ROOT, MODEL_ROOT, ADAPTER_PATH, CASES_PATH
import torch
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration,Qwen2TokenizerFast
from h3_runtime.encoder_tp import shard_text_model

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    control = dist.new_group(backend='gloo')
    torch.set_grad_enabled(False)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_ROOT/'text_encoder'),
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(MODEL_ROOT/'tokenizer'),local_files_only=True)
    cases=json.loads((CASES_PATH).read_text())
    references=[]
    if rank==0:
        model.to('cuda')
        for case in cases:
            ids=tokenizer(case['prompt'],add_special_tokens=False,return_tensors='pt').input_ids.cuda()
            output=model.model(input_ids=ids,output_hidden_states=True,use_cache=False)
            references.append(output.hidden_states[50].cpu())
            del output
            print('FULL_ENCODER_REFERENCE',case['id'],flush=True)
        model.to('cpu');gc.collect();torch.cuda.empty_cache()
        torch.save(references,OUT/'encoder_reference_hidden50.pt')
    dist.barrier(group=control)
    sharding=shard_text_model(model.model.language_model)
    model.to('cuda')
    comparisons=[]
    for i,case in enumerate(cases):
        ids=tokenizer(case['prompt'],add_special_tokens=False,return_tensors='pt').input_ids.cuda()
        output=model.model(input_ids=ids,output_hidden_states=True,use_cache=False)
        got=output.hidden_states[50]
        want=references[i].cuda() if rank==0 else torch.empty_like(got)
        dist.broadcast(want,src=0)
        difference=got.float()-want.float()
        relative=float(difference.norm()/want.float().norm().clamp_min(1e-12))
        comparisons.append({'case':case['id'],'shape':list(got.shape),'relative_l2':relative,
            'max_abs':float(difference.abs().max()),'finite':bool(torch.isfinite(got).all()),
            'limit_relative_l2':0.02,'passed':relative<0.02 and bool(torch.isfinite(got).all())})
        del output,got,want,difference
    report={'rank':rank,'sharding':sharding,'comparisons':comparisons,
        'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'status':'passed' if all(c['passed'] for c in comparisons) else 'failed'}
    (OUT/f'encoder_real_parity_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REAL_ENCODER_PARITY',json.dumps(report),flush=True)
    dist.destroy_process_group()
    assert report['status']=='passed'

if __name__=='__main__':main()
