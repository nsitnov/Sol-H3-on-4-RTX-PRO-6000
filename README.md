# Sol-H3 on 4 RTX PRO 6000 GPUs

This repository documents and packages the adaptation that made **MiniMax-H3 / Sol-H3 run cooperatively on four NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB GPUs**. It includes the exact runtime patch, pinned dependencies and model revisions, portable reproduction scripts, and the measurements from the successful run on September 7, 2026.

The original T2V change is **tensor parallelism for the Qwen3-VL text encoder before the pipeline is moved to CUDA**. The later image-conditioned extension distributes whole encoder layers to preserve multimodal numerical fidelity; see [I2V and Ref2VA](docs/IMAGE_TESTS.md). The unmodified loading path exceeded each GPU's memory capacity even when four workers were running. Sol-H3 already distributed DiT sequence computation and VAE work; that alone did not distribute the large encoder weights.

## Measured results

Official four-step FastH3 / SOL-BSA profile, 1344×768, 24 FPS, seed 42, native synchronized audio. Every case had one warmup and three measured requests in a resident engine.

| Test case | Median time to finished MP4 | Ratio vs. historical local 20-step control | Ratio vs. previous vLLM-Omni on 4 GPUs |
|---|---:|---:|---:|
| 5-second dialogue | **12.19 s** | **14.61×** | **2.95×** |
| 10-second long dialogue | **28.23 s** | **19.49×** | **3.11×** |
| 10-second short speech, followed by silence | **28.20 s** | N/A | **3.12×** |

These are ratios between measured configurations, **not an isolated speedup from the encoder patch or from GPU count**. Historical controls used another host and different precision/sampler settings. The local 20-step ComfyUI control is not a measurement of an external production service. The previous vLLM clips contain 120/240 frames; Sol-H3 uses its native 124/243 frames.

**Visual quality and speech fidelity are awaiting human review. The original 10× acceleration target with preserved quality has not been achieved.** The four-step adapter, sparse attention and reduced-precision attention transport are approximate methods. Numerical checks of encoder outputs are useful implementation checks, not human quality acceptance.

## I2V and Ref2VA results

Four additional cases completed: **16 renders, including four warmups and twelve measurements**. All use four GPUs and four DiT forwards with native audio.

| Case | Sol-H3 median to MP4 | Historical vLLM4 | Timing ratio |
|---|---:|---:|---:|
| I2V: chef, 5 seconds | **13.76 s** | 39.16 s | **2.85×** |
| I2V: chef, 10 seconds | **30.74 s** | 92.89 s | **3.02×** |
| Ref2VA: observatory, 5 seconds | **12.49 s** | 61.55 s | **4.93×** |
| Ref2VA: neon street, 10 seconds | **27.66 s** | 153.65 s | **5.56×** |

I2V uses FastH3 with first-frame conditioning; Ref2VA uses its dedicated transformer and LightX four-step adapter. The encoder is distributed by whole layers, with bit-exact full BF16 parity for all four image cases on every rank. Historical vLLM4 used eight steps, different adapters/reference preprocessing and another host. These are configuration timing ratios; human quality review is pending.

[Reproduce the image tests and understand the encoder change](docs/IMAGE_TESTS.md) · [All image measurements](benchmarks/2026-09-07-images/measurements.json) · [Review videos and source images](https://built-journey-toolkit-appendix.trycloudflare.com/sol-h3-image.html) (temporary host).

## Start here

1. [Reproduction guide](docs/REPRODUCE.md): hardware, environment, pinned downloads, patch application, checks, generation and timing.
2. [How the adaptation works](docs/ARCHITECTURE.md): the memory failure, column/row partitioning, hidden-state semantics, and which optimizations belong to upstream Sol-H3.
3. [Benchmark methodology and evidence](docs/BENCHMARKS.md): every repetition, stage timings, validation, limitations and failed attempts.
4. [Eight-GPU forecast](docs/EIGHT_GPU.md): a transparent model, memory estimate, experimental commands, and latency versus throughput tradeoffs.
5. [Troubleshooting](docs/TROUBLESHOOTING.md): model loading, OOM, SM120 kernels, synchronization, and MP4 validation.
6. [Package validation](docs/VALIDATION.md): checks actually run while preparing this repository, and remaining validation boundaries.

The practical sequence is:

```bash
git clone https://github.com/nsitnov/Sol-H3-on-4-RTX-PRO-6000.git
cd Sol-H3-on-4-RTX-PRO-6000
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.lock.txt
python scripts/setup_source.py
hf auth login
python scripts/download_models.py
```

**Continue with the [reproduction guide](docs/REPRODUCE.md) before launching workers.** It covers system packages, disk/RAM requirements, the three readiness checks and the benchmark command. Model download is approximately 145.5 GB including the adapter. No model weights are stored in this repository.

## What is included

| Path | Purpose |
|---|---|
| `patches/sol-h3-encoder-tp.patch` | Exact patch used by the original T2V campaign |
| `patches/sol-h3-image-encoder.patch` | Cumulative image patch with whole-layer encoder distribution |
| `requirements.lock.txt` | Observed Python package versions and pinned Git dependencies |
| `configs/weights.json` | Base/adapter revisions, file sizes and checksums |
| `configs/cases.json` | Exact English benchmark prompts |
| `scripts/` | Portable setup, download, probe, parity, generation and reporting tools |
| `benchmarks/2026-09-07/` | All 12 recorded renders' metadata and all-rank numerical checks |
| `benchmarks/2026-09-07-images/` | All 16 image renders, multimodal numerical checks and validation |
| `archive/benchmark_sol_h3_20260907.py` | Exact original benchmark runner for provenance; see [archive notes](archive/README.md) |

The portable scripts retain the measured runtime patch and generation settings, but replace machine-specific paths and generalize rank counts for an experimental eight-GPU run. The archived runner is the source of the published measurements. Neither a clean-host reinstall nor an eight-GPU full generation is claimed as a completed test.

## Eight GPUs: expectation, not a benchmark

A stage-based planning model gives approximately **8.2–10.2 seconds for the original T2V 5-second case** and **18.8–23.3 seconds for the original T2V long 10-second case** on eight comparable cards, under explicit assumptions about compute scaling and communication overhead. Those scenarios correspond to roughly **1.2–1.5× lower latency than our four-GPU run**, not another 5–10× improvement. Poor topology can eliminate the gain or make eight GPUs slower.

An idealized model with twice the denoise/VAE speed and no extra communication gives 6.45/14.77 seconds; this is a model reference, not an expected result or a guaranteed lower bound. See the [derivation and experimental procedure](docs/EIGHT_GPU.md). Two independent four-GPU workers may be preferable for serving concurrent requests.

## Attribution

The base model is [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). The inference stack and its existing optimizations are from [NVIDIA's Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/). The four-step adapter is [FastVideo's FastH3 dense-datafree preview](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA). The encoder partitioning approach was informed by [vLLM-Omni's MiniMax-H3 encoder](https://github.com/vllm-project/vllm-omni/blob/ae70479619b9c80b37f0ce29558a72913bc46056/vllm_omni/diffusion/models/minimax_h3/encoder.py).

This repository is an independent adaptation and replication record. It does not replace the upstream projects, their licenses, or the model terms. See [third-party notes](THIRD_PARTY_NOTICES.md).
