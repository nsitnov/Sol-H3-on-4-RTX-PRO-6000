"""Launch one exclusive, logged four-GPU campaign (eight GPUs are experimental)."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from common import ROOT, OUT, SANA_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if not args.attempt.replace('_', '').replace('-', '').isalnum():
        raise SystemExit('Use letters, digits, underscores or hyphens in --attempt')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / args.attempt).exists():
        raise SystemExit('Attempt already exists; choose a new name')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise SystemExit('GPU workers exist on this host; this launcher requires an exclusive host')
    for rank in range(args.gpus):
        for name in ('sm120_probe', 'encoder_real_parity'):
            report = json.loads((OUT / f'{name}_rank{rank}.json').read_text())
            assert report['status'] == 'passed', (name, rank)
        parity = json.loads((OUT / f'encoder_real_parity_rank{rank}.json').read_text())
        assert parity['sharding']['world_size'] == args.gpus, 'Repeat parity for this GPU count'
        tiny = json.loads((OUT / f'encoder_tp_test_rank{rank}.json').read_text())
        assert len(tiny) == 2
        for check in tiny:
            assert check['sharding']['world_size'] == args.gpus
            limit = 0.02 if check['dtype'] == 'torch.bfloat16' else 1e-5
            assert all(error < limit for error in check['relative_l2_by_hidden_state'])
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', ','.join(str(i) for i in range(args.gpus)))
    env.setdefault('OMP_NUM_THREADS', '4')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', H3_ENCODER_TP='1',
               H3_SOL_ATTEMPT=args.attempt, H3_ULYSSES_COMM_DTYPE='int8_qkv',
               H3_ULYSSES_INT8_SCOPE='all', H3_ULYSSES_OUTPUT_DTYPE='fp8', SOL_ATTN_STRICT='1')
    subprocess.run([sys.executable, str(ROOT / 'scripts/preflight.py'), '--gpus', str(args.gpus)], env=env, check=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={args.gpus}', str(ROOT / 'scripts/benchmark.py')]
    patch = subprocess.check_output(['git', '-C', str(SANA_ROOT), 'diff'])
    launch = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': command, 'gpu_count': args.gpus,
              'environment': {k: v for k, v in env.items() if k.startswith('H3_') or k in
                              ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'SOL_ATTN_STRICT')},
              'runner_sha256': hashlib.sha256((ROOT / 'scripts/benchmark.py').read_bytes()).hexdigest(),
              'upstream_diff_sha256': hashlib.sha256(patch).hexdigest(),
              'eight_gpu_status': 'experimental_unmeasured' if args.gpus == 8 else 'four_gpu_replication'}
    (OUT / f'{args.attempt}-launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    (OUT / f'{args.attempt}-upstream.patch').write_bytes(patch)
    print(f'Worker log: {OUT / (args.attempt + ".log")}', flush=True)
    with (OUT / f'{args.attempt}.log').open('w') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise SystemExit(process.returncode)
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize.py'), str(OUT / args.attempt)], check=True)

if __name__ == '__main__':
    main()
