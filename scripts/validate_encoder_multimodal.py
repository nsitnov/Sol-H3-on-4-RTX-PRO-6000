"""Compare the exact I2V/Ref2VA presentations through full BF16 and distributed encoders."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageOps

from common import ROOT, OUT, MODEL_ROOT
from image_common import IMAGE_CASES_PATH
MODEL = MODEL_ROOT
import torch
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration, Qwen2TokenizerFast, Qwen3VLProcessor
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference, before_encoder
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep, MiniMaxH3Ref2VATextEncoderStep
from h3_runtime.encoder_tp import shard_text_model
from h3_runtime.encoder_pp import distribute_text_layers
from h3_runtime.engine import _resolve_reference_image_size

def encode(components, case):
    if case['sol_task']=='i2v':
        path=ROOT/case['inputs'][0]['path']
        with Image.open(path) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
        image=before_encoder.prepare_keyframe_image(image,768,1344,stretch=True)
        return MiniMaxH3TextEncoderStep.encode_prompt(components,case['prompt'],[image],
                    device=torch.device('cuda'),dtype=torch.bfloat16)
    references=[MiniMaxH3Reference(image=str(ROOT/item['path'])) for item in case['inputs']]
    prepared,_=before_encoder.MiniMaxH3Ref2VASetupStep.prepare_references(
        components,references,{5:124,10:243}[case['seconds']])
    return MiniMaxH3Ref2VATextEncoderStep.encode_prompt(components,case['prompt'],prepared,
                device=torch.device('cuda'),dtype=torch.bfloat16)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    control=dist.new_group(backend='gloo')
    torch.set_grad_enabled(False)
    before_encoder.resolve_reference_image_size=lambda width,height: _resolve_reference_image_size(width,height,mode='match')
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL/'text_encoder'),
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    components=SimpleNamespace(text_encoder=model,
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(MODEL/'tokenizer'),local_files_only=True),
        processor=Qwen3VLProcessor.from_pretrained(str(MODEL/'processor'),local_files_only=True),
        audio_sampling_rate=32000)
    cases=json.loads((IMAGE_CASES_PATH).read_text())
    for case in cases:
        for item in case['inputs']:
            assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    references=[]
    if rank==0:
        cached=os.environ.get('H3_PARITY_REFERENCE_DIR')
        if cached:
            reference_dir=Path(cached)
            provenance=json.loads((reference_dir/'encoder_multimodal_parity_rank0.json').read_text())
            for case,old in zip(cases,provenance['comparisons'],strict=True):
                assert case['id']==old['case'] and case['inputs']==old['inputs']
                assert hashlib.sha256(case['prompt'].encode()).hexdigest()==old['prompt_sha256']
            references=torch.load(reference_dir/'multimodal_reference_hidden50.pt',weights_only=True)
            print('REUSED_FULL_BF16_REFERENCE',reference_dir,flush=True)
        else:
            model.to('cuda')
            for case in cases:
                hidden,tags=encode(components,case)
                references.append((hidden.cpu(),tags.cpu()))
                print('FULL_MULTIMODAL_REFERENCE',case['id'],list(hidden.shape),flush=True)
                del hidden,tags
            model.to('cpu');gc.collect();torch.cuda.empty_cache()
            torch.save(references,OUT/'multimodal_reference_hidden50.pt')
    dist.barrier(group=control)
    sharding=(distribute_text_layers if os.environ.get('H3_ENCODER_PP')=='1' else shard_text_model)(model.model.language_model)
    model.to('cuda')
    comparisons=[]
    for i,case in enumerate(cases):
        got,tags=encode(components,case)
        want=references[i][0].cuda() if rank==0 else torch.empty_like(got)
        want_tags=references[i][1].cuda() if rank==0 else torch.empty_like(tags,device='cuda')
        dist.broadcast(want,src=0);dist.broadcast(want_tags,src=0)
        difference=got.float()-want.float()
        relative=float(difference.norm()/want.float().norm().clamp_min(1e-12))
        finite=bool(torch.isfinite(got).all());tags_match=bool(torch.equal(tags.cuda(),want_tags))
        comparison={'case':case['id'],'shape':list(got.shape),'relative_l2':relative,
            'max_abs':float(difference.abs().max()),'finite':finite,'tags_match':tags_match,
            'limit_relative_l2':0.02,'passed':relative<0.02 and finite and tags_match,
            'prompt_sha256':hashlib.sha256(case['prompt'].encode()).hexdigest(),
            'inputs':case['inputs'],'resize_profile':'i2v_target_canvas' if case['sol_task']=='i2v' else 'ref2va_match'}
        comparisons.append(comparison)
        print('MULTIMODAL_PARITY_CASE',rank,case['id'],relative,flush=True)
        del got,tags,want,want_tags,difference
    report={'rank':rank,'sharding':sharding,'comparisons':comparisons,
        'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'status':'passed' if all(c['passed'] for c in comparisons) else 'failed'}
    (OUT/f'encoder_multimodal_parity_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MULTIMODAL_ENCODER_PARITY',rank,report['status'],flush=True)
    dist.destroy_process_group()
    assert report['status']=='passed'

if __name__=='__main__':main()
