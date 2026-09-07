"""Multi-rank SM120 readiness gate for the official SOL/BSA attention path."""
import json
import os
import sys
import traceback
from pathlib import Path

from common import ROOT, OUT, SANA_ROOT, MODEL_ROOT, ADAPTER_PATH, CASES_PATH
import torch
import torch.distributed as dist

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    report = {'rank':rank, 'capability':torch.cuda.get_device_capability(),
              'torch':torch.__version__, 'status':'running'}
    try:
        # Exact all-to-all transport first, independent of the approximate kernels.
        world = dist.get_world_size()
        source = torch.arange(64 * world, device='cuda', dtype=torch.float32) + rank*1000
        output = torch.empty_like(source)
        dist.all_to_all_single(output, source)
        expected = torch.cat([torch.arange(rank*64, (rank+1)*64, device='cuda')+r*1000
                              for r in range(world)])
        assert torch.equal(output, expected)
        report['nccl_all_to_all_exact'] = True
        from h3_runtime.sparse_attention import H3SparseAttention
        # Exercise the actual upstream BSA numerical gate with all blocks selected.
        attention = H3SparseAttention(backend='sol_bsa', gate=True)
        torch.manual_seed(42)
        q, k, v = [torch.randn(1,256,56//world,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        attention._run_bsa_gate(q,k,v)
        report['bsa_gate'] = attention.gate_stats
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    target = OUT/f'sm120_probe_rank{rank}.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    dist.destroy_process_group()
    if report['status'] != 'passed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
