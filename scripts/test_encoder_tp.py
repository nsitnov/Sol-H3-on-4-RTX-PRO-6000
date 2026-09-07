"""Compare full Qwen attention/MLP hidden states with tensor-parallel sharding."""
import copy
import json
import os
import sys
from pathlib import Path
from common import ROOT, OUT, SANA_ROOT, MODEL_ROOT, ADAPTER_PATH, CASES_PATH
import torch
import torch.distributed as dist
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from h3_runtime.encoder_tp import shard_text_model

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    config=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=3,num_attention_heads=8 if dist.get_world_size()==4 else 16,num_key_value_heads=dist.get_world_size(),head_dim=16,
        rope_parameters={'rope_type':'default','rope_theta':1000000.,'mrope_section':[2,3,3]})
    config._attn_implementation='sdpa'
    reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(42)
        full=Qwen3VLTextModel(config).eval().to(device='cuda',dtype=dtype)
        sharded=copy.deepcopy(full)
        shard_report=shard_text_model(sharded)
        ids=torch.arange(33,device='cuda').reshape(1,-1)
        with torch.no_grad():
            want=full(input_ids=ids,use_cache=False,output_hidden_states=True)
            got=sharded(input_ids=ids,use_cache=False,output_hidden_states=True)
        errors=[]
        for x,y in zip(want.hidden_states,got.hidden_states):
            rel=float((x.float()-y.float()).norm()/x.float().norm().clamp_min(1e-12))
            errors.append(rel)
            assert rel < (0.02 if dtype==torch.bfloat16 else 1e-5), errors
        reports.append({'dtype':str(dtype),'relative_l2_by_hidden_state':errors,'sharding':shard_report})
    (OUT/f'encoder_tp_test_rank{rank}.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('ENCODER_TP_PASS',rank,reports,flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
