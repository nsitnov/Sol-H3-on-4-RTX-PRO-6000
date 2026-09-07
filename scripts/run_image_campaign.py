"""Run each image mode on four owned GPUs with immutable attempt records."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, OUT, SANA_ROOT
from image_common import IMAGE_CASES_PATH, REF_MODEL_ROOT, REF_ADAPTER_PATH

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--task',choices=('i2v','ref2va'),required=True)
    args=parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Invalid attempt name')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU workers exist; refusing overlap')
    for i in range(4):
        parity=json.loads((OUT/f'encoder_multimodal_parity_rank{i}.json').read_text())
        assert parity['status']=='passed'
        assert parity['sharding']['strategy']=='layer_pipeline'
    for case in json.loads(IMAGE_CASES_PATH.read_text()):
        for item in case['inputs']:
            assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    for i in range(4):
        assert json.loads((OUT/f'sm120_probe_rank{i}.json').read_text())['status']=='passed'
    if args.task=='ref2va':
        assert (REF_MODEL_ROOT/'transformer_ref/config.json').exists() and REF_ADAPTER_PATH.exists()
    target=OUT/args.attempt;target.mkdir(exist_ok=False)
    env=os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3')
    env.update(OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',H3_ENCODER_TP='0',H3_ENCODER_PP='1',H3_SOL_TASK=args.task,H3_SOL_ATTEMPT=args.attempt,
        H3_ULYSSES_COMM_DTYPE='int8_qkv',H3_ULYSSES_INT8_SCOPE='all',H3_ULYSSES_OUTPUT_DTYPE='fp8',SOL_ATTN_STRICT='1')
    subprocess.run([sys.executable,str(ROOT/'scripts/preflight.py'),'--gpus','4'],env=env,check=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(ROOT/'scripts/benchmark_image.py')]
    patch=subprocess.check_output(['git','-C',str(SANA_ROOT),'diff'])
    runner=(ROOT/'scripts/benchmark_image.py').read_bytes()
    launch={'command':command,'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
        'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','H3_ENCODER_TP','H3_ENCODER_PP','H3_SOL_TASK','H3_SOL_ATTEMPT',
        'H3_ULYSSES_COMM_DTYPE','H3_ULYSSES_INT8_SCOPE','H3_ULYSSES_OUTPUT_DTYPE','SOL_ATTN_STRICT')},
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runner_sha256':hashlib.sha256(runner).hexdigest(),'patch_sha256':hashlib.sha256(patch).hexdigest()}
    (target/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    (target/'benchmark_runner.py').write_bytes(runner)
    (target/'vendor_changes.patch').write_bytes(patch)
    with (target/'server.log').open('w') as log,(target/'memory.jsonl').open('w') as memory:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('SOL_IMAGE_WORKERS_STARTED',args.task,process.pid,flush=True)
        while process.poll() is None:
            stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            memory.write(json.dumps({'time':time.time(),'gpus':stats})+'\n');memory.flush()
            time.sleep(5)
    summary={'task':args.task,'returncode':process.returncode,
        'measured_records':len(list(target.glob('*/repeat*/result.json'))),
        'warmup_records':len(list(target.glob('*/warmup/result.json'))),
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (target/'campaign.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SOL_IMAGE_CAMPAIGN_END',json.dumps(summary),flush=True)
    if process.returncode==0:
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_image.py'),str(target)],check=True)
    return process.returncode

if __name__=='__main__':sys.exit(main())
