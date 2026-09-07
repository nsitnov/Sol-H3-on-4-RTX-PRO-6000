# Troubleshooting and limits

| Symptom | What to inspect | Resolution or interpretation |
|---|---|---|
| OOM at `pipe.to` on every rank | Is the patch present? Is `H3_ENCODER_TP=1` set in every worker? | Look for the encoder TP report before `.to(cuda)`. TP4 should retain about 18.54 GiB of total encoder parameters per rank. Four processes alone do not pool memory. |
| OOM despite TP | Other GPU processes, exact 96GB Blackwell SKU, model partition, peak load/compile memory | Start on idle cards and use the pinned T2V components. The measured warmup used almost all available GPU memory. The patch does not validate 48GB cards or arbitrary workloads. |
| Process killed without a CUDA OOM | System RAM, cgroup/container memory cap, kernel OOM log | Each process initially loads CPU components. Increase available host memory or design a different checkpoint loader; an undocumented loader change is a new configuration. |
| Offline component lookup fails | Local model layout and patch in `engine.py` | Stage all 61 selected files and pass the local root to `load_components`. Complete downloads before enabling offline mode. |
| Adapter download returns 401/404 | Actual repository name and local HF access | Use `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` at the pinned revision. Check any owner access requirements. Do not substitute a similarly named adapter. |
| Missing MiniMax-H3 classes in Diffusers | Installed Diffusers source revision | Install the pinned commit from the lock; a generic PyPI release may have another API. |
| BSA kernel import/JIT failure | Torch CUDA build, Triton, cuDNN Frontend/CUTLASS packages, driver, architecture | Restore the pinned environment and run the small probe. A passing import is insufficient; SM120 must execute the gate. |
| NCCL hangs or errors | Matching rank count, GPU ownership, topology, shared-memory limits, process logs | Every rank must enter the same collectives. Use `NCCL_DEBUG=INFO` temporarily and retain the logs. Do not disable transports blindly or stop unrelated workers. |
| Valid MP4 exists but job appears stuck after saving | Whether benchmark control uses Gloo | The intermediate attempt stalled after save. The successful benchmark moved CPU barriers/object-gather to Gloo while keeping GPU math on NCCL. |
| `ffprobe` lacks `nb_frames` | Is this fragmented MP4? | Count decoded frames with `ffprobe -count_frames`; read `nb_read_frames` and run a full AV decode. |
| Different MP4 hashes with seed 42 | Actual repeated outputs | This occurred in the completed campaign. Seed equality is not a bitwise-determinism guarantee with these kernels and collectives. |
| Numerical gate passes but speech contains extra words | Human review of original audio and silence | Numerical parity is not speech-fidelity acceptance. ASR WER=0 alone also does not establish acceptance. |
| Eight ranks fail although four passed | Fresh all-rank probe/parity, seven-head DiT shape, topology and memory | Eight-GPU support is experimental here. Preserve the failure and do not report a forecast as a successful run. |

For a reproducible issue report, include the repository/upstream revisions, Python package freeze, GPU model/driver, topology, exact command and selected environment settings, and the failing stage. Include the associated rank audits and exception, but keep authentication tokens out of logs and reports.

The benchmark launcher refuses an existing attempt directory and existing GPU workers. It never kills a service. Use a new attempt name after diagnosing a failed run; keep the earlier records so successful samples are not selectively combined across attempts.

