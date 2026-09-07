"""Inspect availability without loading the model or stopping other processes."""
import argparse
import json
import shutil
import subprocess
from common import SANA_ROOT, SANA_REVISION, MODEL_ROOT, ADAPTER_PATH, RUNTIME_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    import torch
    for executable in ('git', 'ffmpeg', 'ffprobe', 'nvidia-smi'):
        if not shutil.which(executable):
            raise SystemExit(f'Missing executable: {executable}')
    if torch.cuda.device_count() != args.gpus:
        raise SystemExit(f'Expose exactly {args.gpus} GPUs through CUDA_VISIBLE_DEVICES')
    if subprocess.check_output(['git', '-C', str(SANA_ROOT), 'rev-parse', 'HEAD'], text=True).strip() != SANA_REVISION:
        raise SystemExit('Upstream revision differs from the tested revision')
    if not (RUNTIME_ROOT / 'h3_runtime/encoder_tp.py').is_file():
        raise SystemExit('Encoder TP patch is missing')
    for path in (MODEL_ROOT / 'model_index.json', ADAPTER_PATH):
        if not path.is_file():
            raise SystemExit(f'Missing model asset: {path}')
    inventory = []
    for i in range(args.gpus):
        prop = torch.cuda.get_device_properties(i)
        if (prop.major, prop.minor) != (12, 0) or prop.total_memory < 90 * 1024**3:
            raise SystemExit(f'GPU {i} is outside the tested SM120 / 96GB hardware profile')
        inventory.append({'rank': i, 'name': prop.name, 'memory_bytes': prop.total_memory})
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpus': inventory}, indent=2))
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], check=True)
    print('Review the GPU process list before launching. Eight-GPU execution is experimental.' if args.gpus == 8
          else 'Review the GPU process list before launching.')

if __name__ == '__main__':
    main()

