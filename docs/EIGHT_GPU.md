# What to expect on eight RTX PRO 6000 GPUs

These numeric scenarios use the original T2V measurements. The [image-conditioned campaign](IMAGE_TESTS.md#eight-gpus) uses a different encoder distribution and additional reference encoding; its eight-GPU latency has not been measured or projected with these same numbers.

**This is a forecast, not a measured eight-GPU result.** The completed experiment used four cards. No eight-GPU RTX generation, latency, output quality or numerical parity is claimed here.

There are two different goals: reduce the latency of one clip by cooperating across eight GPUs, or increase throughput by running two independent four-GPU replicas. They require different benchmarks.

## Feasibility of one eight-GPU worker

The pinned Sol-H3 engine accepts 1, 2, 4 or 8 ranks and chooses Ulysses degree equal to the world size. The new encoder sharding function uses the actual process-group size and requires divisibility of attention heads, KV heads and MLP channels.

The pinned model dimensions satisfy the eight-rank arithmetic:

| Dimension | Total | Per GPU at 4 ranks | Per GPU at 8 ranks |
|---|---:|---:|---:|
| Encoder query heads | 64 | 16 | 8 |
| Encoder KV heads | 8 | 2 | 1 |
| Encoder MLP intermediate channels | 25,600 | 6,400 | 3,200 |
| DiT attention heads after the Ulysses exchange | 56 | 14 | 7 |

This establishes dimension compatibility only. Seven DiT heads per rank, different sequence partitions, SM120 kernel specialization, VAE tile distribution and eight-rank collectives still need real tests. The small probe in this repository uses `56 / world_size` heads, so an eight-rank invocation exercises the relevant seven-head small shape.

The simple encoder partitioning is bounded by the eight KV heads. More than eight ranks would require another strategy, such as KV-head replication or separate process groups, and is outside this patch.

## Weight-memory estimate

From the measured full and four-rank encoder sizes, the encoder can be represented as:

```text
M_encoder(p) = R + S / p
R =  4,303,536,608 bytes  (replicated parameters)
S = 62,411,243,520 bytes  (partitioned parameters)
```

| Ranks | Encoder weights per GPU | Status |
|---|---:|---|
| 1 | 62.13 GiB | full-encoder accounting |
| 4 | 18.54 GiB | measured |
| 8 | **11.27 GiB** | calculated |

Eight ranks would therefore save another approximately 7.27 GiB of encoder weights per GPU relative to TP4. The approximate initial total weight lower bound would fall to 78.1 GiB per rank. DiT weight storage is largely replicated in this setup, so it does not simply halve when the GPU count doubles.

Peak memory includes much more than these weights. The full CPU model is also initially loaded by each process, so doubling ranks can increase host-memory pressure. Plan a large-memory host and measure its peak; no minimum RAM figure for eight ranks has been validated.

## Latency model

For each case, start from the measured four-GPU full-request median `T4`, denoise median `D4`, and video-decode median `V4`. Define the residual `O4 = T4 - D4 - V4` and model:

```text
T8 = O4 + D4 / sD + V4 / sV + delta
```

`sD` and `sV` are assumed four-to-eight-GPU speedups of denoising and video decode. `delta` represents additional communication/synchronization overhead. The residual holds text encoding, audio decode, MP4 saving and other overhead at their four-GPU aggregate value. It is a modeling residual, not an independently measured stage; medians of stages need not sum to the median of the total.

Text encoding is only about 0.05–0.06 seconds after warmup, so optimizing it further cannot produce a major end-to-end speedup. It was the crucial memory fix, while denoising still consumes roughly 80–84% of full-request time.

### Explicit scenarios

These assumptions are chosen planning scenarios, not a fitted scaling curve or statistical confidence interval.

| Scenario | Denoise speedup `sD` | Video-decode speedup `sV` | Extra overhead `delta`, 5s / 10s |
|---|---:|---:|---:|
| Conservative improvement | 1.3× | 1.2× | 0.50 / 1.00 s |
| Balanced improvement | 1.6× | 1.5× | 0.25 / 0.50 s |
| Idealized compute scaling | 2.0× | 2.0× | 0 / 0 s |

| Case | Measured 4 GPUs | Conservative 8-GPU scenario | Balanced 8-GPU scenario | Idealized model |
|---|---:|---:|---:|---:|
| 5-second dialogue | 12.19 s | 10.15 s | 8.20 s | 6.45 s |
| 10-second long dialogue | 28.23 s | 23.23 s | 18.77 s | 14.77 s |
| 10-second short speech | 28.20 s | 23.20 s | 18.74 s | 14.74 s |

The conservative-to-balanced scenarios suggest roughly **1.2–1.5× lower single-request latency**. Treat approximately 8–10 seconds for 5-second output and 19–23 seconds for 10-second output as planning examples under those assumptions. They are not guaranteed ranges: an eight-GPU topology with poor cross-socket bandwidth could yield little improvement or a regression. The idealized scenario is not a physical lower bound; different kernels or bottlenecks could change it.

Recalculate these values from the recorded stage medians with:

```bash
python scripts/forecast_eight_gpu.py
```

The original [upstream Sol-H3 README at the pinned revision](https://github.com/NVlabs/Sana/blob/2936c47637380842aaa4a4488fac5006cc542b70/models/minimax_h3/Sol-H3/README.md) reports four-to-eight B300 pipeline improvements of about 1.77× for 5 seconds and 1.87× for 10 seconds. Those observations show that upstream has an eight-rank path, but they are not RTX forecasts: B300 uses a different accelerator/interconnect, the prompt differs, and those timings exclude MP4 saving. Do not divide or multiply our RTX timings by the B300 ratios and present the result as a measurement.

## Why doubling cards need not halve latency

Ulysses performs all-to-all exchanges, while the encoder performs two all-reduces per layer. Eight GPUs increase collective participants and can cross additional PCIe bridges or NUMA nodes. Sequence work per rank shrinks, but launch overheads, synchronization, communication and rank-zero MP4 encoding do not all shrink with it.

VAE tile distribution can also be uneven or padded. Each GPU's local workload may become too small for the same kernel efficiency. Power limits, concurrent CPU encoding and host-memory bandwidth can change the result even if all eight cards have the same model name.

Record `nvidia-smi topo -m`, peer-access capabilities, NCCL transport information, power limits and clocks. Run small collective tests first, then the complete pipeline. An eight-GPU configuration file alone does not demonstrate scaling.

## Experimental eight-GPU procedure

Complete the [four-GPU setup](REPRODUCE.md) first. On an eight-card host, choose a new result directory and expose exactly those devices:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_RUN_ROOT="$PWD/runs/rtx8-experimental"
mkdir -p "$H3_RUN_ROOT"
python scripts/preflight.py --gpus 8
torchrun --standalone --nproc_per_node=8 scripts/probe.py
torchrun --standalone --nproc_per_node=8 scripts/test_encoder_tp.py
torchrun --standalone --nproc_per_node=8 scripts/validate_encoder_real.py
python scripts/run_campaign.py --gpus 8 --attempt experimental8_a1
```

Keep the same model revisions, patch, adapter, dtype, transport settings, prompts, seed, durations and four actual DiT forwards. The portable scripts use dynamic rank counts; the archived original runner is hard-coded to four and must not be used for this command.

All eight ranks must pass fresh numerical checks. For eight ranks the tiny test uses eight KV heads; the full-checkpoint test remains the essential comparison. Recheck output video/audio and obtain a human speech/quality review because new reduction orders and attention partitions can alter the generated result.

For a fair scaling measurement, run both four- and eight-GPU campaigns on the **same eight-card host**, using a documented four-card subset, separate warmups and the same CPU resources. Report `median(T4) / median(T8)`, all repetitions, per-stage timings and peak memory. The existing four-card historical run alone cannot isolate the effect of doubling GPUs on a new machine.

## Alternative: two independent four-GPU replicas

If throughput matters more than one-request latency, place one four-GPU engine on each suitable GPU group and dispatch separate requests to them. Under ideal isolation and enough queued work, aggregate throughput could approach twice a single four-GPU engine, while each request retains roughly its original 12.2/28.2-second latency.

That is a throughput hypothesis, not an eight-GPU single-clip speedup. CPU MP4 encoding, shared model reads, RAM and PCIe topology can prevent a full 2× gain. The simple `run_campaign.py` launcher intentionally requires an exclusive host; use an allocation-aware service or scheduler for the two-replica experiment, with separate ports, output directories and GPU assignments.

Measure both strategies before choosing a deployment: eight-way cooperation for latency, two four-way replicas for concurrency. No eight-card strategy in this document has yet been benchmarked on RTX PRO 6000.

