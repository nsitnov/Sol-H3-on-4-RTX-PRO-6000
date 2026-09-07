"""Resident official Sol-H3 profile; each case has one warmup and three repeats."""
import hashlib
import faulthandler
import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sol_h3_image_20260907'
TASK = os.environ['H3_SOL_TASK']
MODEL = '/dev/shm/h3-sol-weights/base' if TASK=='i2v' else '/dev/shm/h3-sol-weights/ref2va'
ADAPTER = '/dev/shm/h3-sol-weights/adapter/dense-datafree/adapter_model.safetensors' if TASK=='i2v' else '/dev/shm/h3-sol-weights/ref_adapter/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors'
SAMPLES = OUT/os.environ['H3_SOL_ATTEMPT']
sys.path.insert(0, str(ROOT/'vendor/sol-h3-upstream/models/minimax_h3/Sol-H3'))
import torch
import torch.distributed as dist
from h3_runtime import MiniMaxH3Inference
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    cases = [c for c in json.loads((OUT/'cases.json').read_text()) if c['sol_task']==TASK]
    adapter_sha256=hashlib.sha256(Path(ADAPTER).read_bytes()).hexdigest()
    start = time.perf_counter()
    with MiniMaxH3Inference(MODEL, ADAPTER, attention_backend='sol_bsa', task=TASK,
            reference_image_resize_mode='match') as engine:
        setup = time.perf_counter()-start
        control = dist.new_group(backend='gloo')
        calls, events, stages = [], [], {}
        # Instrument complete pipeline blocks, outside the kernels and compile regions.
        from diffusers.modular_pipelines.minimax_h3 import encoders, decoders, denoise
        text_cls = encoders.MiniMaxH3TextEncoderStep if TASK=='i2v' else encoders.MiniMaxH3Ref2VATextEncoderStep
        image_cls = encoders.MiniMaxH3KeyframeVaeEncoderStep if TASK=='i2v' else encoders.MiniMaxH3Ref2VAReferenceEncoderStep
        denoise_cls = denoise.MiniMaxH3DenoiseStep if TASK=='i2v' else denoise.MiniMaxH3Ref2VADenoiseStep
        for cls, name in ((text_cls,'text_encode'), (image_cls,'image_encode'),
                          (denoise_cls,'denoise'),
                          (decoders.MiniMaxH3VideoDecodeStep,'video_decode'),
                          (decoders.MiniMaxH3AudioDecodeStep,'audio_decode')):
            original = cls.__call__
            def timed(block, *args, _original=original, _name=name, **kwargs):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                value = _original(block,*args,**kwargs)
                torch.cuda.synchronize()
                stages[_name] = stages.get(_name,0.) + time.perf_counter()-begin
                return value
            cls.__call__ = timed
        def pre(*unused):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append([event, None])
            calls.append(time.perf_counter())
        def post(*unused):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[-1][1] = event
        hooks = [engine.transformer.register_forward_pre_hook(pre),
                 engine.transformer.register_forward_hook(post)]
        for case in cases:
            for repeat in range(4):
                target = SAMPLES/case['id']/('warmup' if repeat == 0 else f'repeat{repeat}')
                target.mkdir(parents=True, exist_ok=True)
                calls.clear()
                events.clear()
                stages.clear()
                dist.barrier(group=control)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if engine.is_rank_zero:
                    (OUT/'active_job.json').write_text(json.dumps({'case':case['id'],'repeat':repeat})+'\n')
                    print('SOL_SAMPLE_START',case['id'],repeat,flush=True)
                started = time.perf_counter()
                # Source media reading belongs to the full request; do not cache references.
                request_inputs = {'image':str(ROOT/case['inputs'][0]['path'])} if TASK=='i2v' else {
                    'references':[MiniMaxH3Reference(image=str(ROOT/item['path'])) for item in case['inputs']]}
                result = engine.generate(case['prompt'], duration=case['seconds'], seed=42, **request_inputs)
                pipeline = time.perf_counter()-started
                print('SOL_PIPELINE_DONE',engine.rank,pipeline,flush=True)
                save_started = time.perf_counter()
                if result is not None:
                    assert result.audio is not None, 'Missing audio'
                    result.save(target/'output.mp4')
                    print('SOL_SAVE_DONE',engine.rank,flush=True)
                dist.barrier(group=control)
                total = time.perf_counter()-started
                save_seconds = time.perf_counter()-save_started
                audit = {'rank':engine.rank,'forwards':len(calls),
                         'pipeline_seconds':pipeline,
                         'denoise_interval_seconds':events[0][0].elapsed_time(events[-1][1])/1000,
                         'transformer_forward_seconds':sum(a.elapsed_time(b) for a,b in events)/1000,
                         'max_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'max_reserved_bytes':torch.cuda.max_memory_reserved(),
                         'stage_seconds':dict(stages),
                         'attention':engine.sparse_attention.stats()}
                audits = [None]*4
                print('SOL_AUDIT_BEGIN',engine.rank,flush=True)
                dist.all_gather_object(audits, audit,group=control)
                assert all(a['forwards']==4 for a in audits), audits
                for a in audits:
                    attention=a['attention']
                    expected_sparse=144 if TASK=='i2v' else 200
                    assert attention['sparse_calls']==expected_sparse*attention['requests'], attention
                    assert attention['dense_calls']==(200-expected_sparse)*attention['requests'], attention
                    assert attention['qkv_wire_dtype']=='int8_qkv' and attention['output_wire_dtype']=='fp8'
                    assert set(attention['declined']).issubset({'warmup_step','dense_layer'} if TASK=='i2v' else set())
                if engine.is_rank_zero:
                    probe = subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format',
                                                     '-of','json',str(target/'output.mp4')],text=True)
                    decoded = subprocess.run(['ffmpeg','-v','error','-i',str(target/'output.mp4'),
                                              '-f','null','-'],capture_output=True,text=True)
                    assert decoded.returncode == 0 and not decoded.stderr.strip(), decoded.stderr
                    streams = json.loads(probe)['streams']
                    video = next(s for s in streams if s['codec_type']=='video')
                    assert (video['width'],video['height']) == (1344,768)
                    # Upstream writes fragmented MP4, where nb_frames is absent.
                    assert int(video['nb_read_frames']) == {5:124,10:243}[case['seconds']]
                    record = {'status':'complete','case':case,'repeat':repeat,'warmup':repeat==0,
                        'attempt':SAMPLES.name,'task':TASK,'model_root':MODEL,'adapter':ADAPTER,
                        'adapter_sha256':adapter_sha256,
                        'reference_image_resize_mode':'match' if TASK=='ref2va' else 'target_canvas',
                        'quality':'pending_human_review','world_size':4,'steps':4,'seed':42,
                        'setup_seconds':setup,'pipeline_seconds':result.elapsed_s,
                        'stage_seconds':dict(stages),'mp4_save_seconds':save_seconds,
                        'encoder_tp':engine.encoder_tp_report,
                        'full_request_seconds':total,'rank_audit':audits,
                        'ffprobe':json.loads(probe),'full_av_decode_passed':True,
                        'sha256':hashlib.sha256((target/'output.mp4').read_bytes()).hexdigest(),
                        'prompt_sha256':hashlib.sha256(case['prompt'].encode()).hexdigest(),
                        'profile':'Official task-specific four-step SOL/BSA BF16 with INT8 QKV and FP8 output transport; encoder distributed by whole layers across four GPUs',
                        'source_revision':subprocess.check_output(['git','-C',str(ROOT/'vendor/sol-h3-upstream'),
                                                                  'rev-parse','HEAD'],text=True).strip()}
                    (target/'result.json').write_text(json.dumps(record,indent=2)+'\n')
                    print('SOL_RESULT',json.dumps({'case':case['id'],'repeat':repeat,'seconds':total}),flush=True)
                del result
                dist.barrier(group=control)
        for hook in hooks:
            hook.remove()
        if engine.is_rank_zero:
            (OUT/'active_job.json').write_text('null\n')

if __name__ == '__main__':
    main()
