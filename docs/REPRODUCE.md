# Reproduce the four-GPU experiment

Run commands from this repository's root in Bash. This guide downloads the original model components and applies the exact small Sol-H3 patch. It does not require the author's workspace, ComfyUI, a vLLM installation, private caches, or the temporary review website.

## Hardware and system requirements

The tested devices were **four RTX PRO 6000 Blackwell Server Edition 96GB cards, SM120**, with driver 595.91.07. This is not the older RTX 6000 Ada 48GB or Quadro RTX 6000. The host reported Ubuntu 24.04.3, Python 3.12.3, 256 logical CPUs and approximately 1.48 TiB of system RAM. The successful campaign used all four cards for a single request.

Plan at least 250 GB of free persistent disk space for the selected checkpoints, adapter, Python environment, source, compilation caches and outputs. The selected model components total 144,051,143,011 bytes; the adapter adds 1,485,626,152 bytes. More space is needed if other model partitions or separate Hub caches are downloaded.

The loader initially constructs full CPU-side components in every process before sharding. **A minimum host-RAM requirement was not established.** About 1 TiB is a conservative planning provision for a four-rank reproduction, not a verified minimum. Monitor available RAM and container/cgroup limits during loading. The original host's large RAM capacity must not be omitted from hardware comparisons. This guide uses disk-backed model files; using `/dev/shm` additionally consumes system RAM and is not required for warm inference.

The CUDA 13 wheel requires a compatible NVIDIA driver. NVIDIA lists the 580 driver family as the CUDA 13.x minor-compatibility floor, but PTX/JIT features can require newer drivers; the tested driver was 595.91.07. Consult [NVIDIA's compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) and use the tested version or a compatible newer driver when reproducing.

Inspect devices and topology before launching:

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
df -h .
free -h
```

The observed GPU topology was:

| | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| GPU0 | X | PIX | NODE | SYS |
| GPU1 | PIX | X | NODE | SYS |
| GPU2 | NODE | NODE | X | SYS |
| GPU3 | SYS | SYS | SYS | X |

`PIX` crosses at most one PCIe bridge; `NODE` crosses host bridges within a NUMA node; `SYS` also crosses the CPU socket interconnect. There were no NVLink entries. This topology matters to all-to-all/all-reduce latency.

## 1. Install system packages and Python dependencies

For Ubuntu with administrative access:

```bash
sudo apt-get update
sudo apt-get install -y git python3.12-venv python3.12-dev build-essential ffmpeg
git clone https://github.com/nsitnov/Sol-H3-on-4-RTX-PRO-6000.git
cd Sol-H3-on-4-RTX-PRO-6000
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.lock.txt
python -m pip check
```

The lock records the environment actually used, including PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130, torchaudio 2.10.0+cu130, Triton 3.6.0 and Transformers 5.8.1. Two dependencies are pinned by source commit:

| Dependency | Commit |
|---|---|
| Diffusers | `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` |
| NVIDIA cuDNN Frontend | `29106622617bfd9031a53099a6fbbc5e74a474e9` |

The package list is an observed freeze, not a hermetic container image or an assertion that all future package indexes will retain every wheel. Preserve a local wheel/build cache for long-term reproduction. If installation differs, save `python -m pip freeze` and report the difference rather than calling the environment identical.

Check the imports and device capability:

```bash
python -c 'import torch, triton, transformers, diffusers; print(torch.__version__, torch.version.cuda, triton.__version__, transformers.__version__, diffusers.__version__); print(torch.cuda.get_device_capability())'
ffmpeg -version
ffprobe -version
```

## 2. Fetch and patch the exact Sol-H3 source

```bash
python scripts/setup_source.py
git -C vendor/Sana rev-parse HEAD
git -C vendor/Sana diff --stat
```

Expected upstream commit: `2936c47637380842aaa4a4488fac5006cc542b70`. The setup script fetches that commit into a **new** `vendor/Sana` directory and applies [the exact patch](../patches/sol-h3-encoder-tp.patch). It refuses to overwrite an existing source directory. Use a new `SANA_ROOT` if needed.

Expected patch SHA256:

```text
af460087a01232e65c2112b02f108f9e1f098334e012ae5d5b0affeeaff479e1
```

For manual patching of a clean checkout at the pinned revision:

```bash
git -C vendor/Sana apply --check ../../patches/sol-h3-encoder-tp.patch
git -C vendor/Sana apply ../../patches/sol-h3-encoder-tp.patch
```

Do not run these two manual commands after `setup_source.py`: it already applied the patch. The patch changes only the local component loading argument, adds the encoder TP hook before `.to(cuda)`, and adds `encoder_tp.py`.

## 3. Download the pinned original weights

Access the [MiniMax-H3 model repository](https://huggingface.co/MiniMaxAI/MiniMax-H3) and complete any access/terms flow required by its owner, then authenticate locally:

```bash
hf auth login
python scripts/download_models.py
```

The downloader selects the official Diffusers T2V components, not all alternative partitions in the repository. It verifies file sizes, SHA256 for LFS model files, Git blob identities for small configuration files, and the adapter SHA256. Verification reads the full checkpoints and can take time.

| Asset | Repository | Revision |
|---|---|---|
| Base | `MiniMaxAI/MiniMax-H3` | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Adapter | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` | `f509e629374cac104e7f62daecce6d1488a3041d` |

Adapter file: `dense-datafree/adapter_model.safetensors`; SHA256:

```text
4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12
```

The doubled `FastVideo` in the adapter repository name is intentional. The old name in the pinned upstream downloader failed to resolve in the original experiment. This repository's downloader uses the verified published name.

Default layout:

```text
models/
  MiniMax-H3/
    model_index.json
    modular_model_index.json
    transformer/
    text_encoder/
    vae/
    audio_vae/
    tokenizer/
    processor/
    scheduler/
    audio_scheduler/
  FastH3/dense-datafree/adapter_model.safetensors
```

The original campaign reused 14 identical encoder checkpoint shards from an existing local cache. This guide downloads them directly, so replication does not depend on that cache. No conversion of a ComfyUI checkpoint is necessary.

## 4. Select paths, devices and the exact runtime profile

All paths below have repository-relative defaults. Override them only to use another storage location:

```bash
export SANA_ROOT="$PWD/vendor/Sana"
export H3_MODEL_ROOT="$PWD/models/MiniMax-H3"
export H3_ADAPTER_PATH="$PWD/models/FastH3/dense-datafree/adapter_model.safetensors"
export H3_RUN_ROOT="$PWD/runs/rtx4"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export H3_ENCODER_TP=1
export H3_ULYSSES_COMM_DTYPE=int8_qkv
export H3_ULYSSES_INT8_SCOPE=all
export H3_ULYSSES_OUTPUT_DTYPE=fp8
export SOL_ATTN_STRICT=1
mkdir -p "$H3_RUN_ROOT"
python scripts/preflight.py --gpus 4
```

Set offline flags only after the model download is complete. The source patch is necessary in addition to the environment variable: `H3_ENCODER_TP=1` has no effect on an unpatched engine. Keep all four ranks resident and make all ranks execute the same sequence of requests and collectives.

## 5. Run the readiness and numerical gates

Use the GPUs exclusively while running these checks. The first two tests do not load the full checkpoint. The third loads the real encoder and creates a full BF16 reference on rank zero before sharding:

```bash
torchrun --standalone --nproc_per_node=4 scripts/probe.py
torchrun --standalone --nproc_per_node=4 scripts/test_encoder_tp.py
torchrun --standalone --nproc_per_node=4 scripts/validate_encoder_real.py
```

Expect one result JSON per rank in `H3_RUN_ROOT`:

| Check | Expected result |
|---|---|
| `sm120_probe_rankN.json` | exact NCCL all-to-all and upstream small BSA gate pass |
| `encoder_tp_test_rankN.json` | every tiny-model hidden state below the FP32/BF16 tolerance |
| `encoder_real_parity_rankN.json` | all three prompts finite and relative L2 below 0.02 at hidden state 50 |

The full check saves `encoder_reference_hidden50.pt` locally for debugging. Do not treat the 2% numerical gate as a visual-quality metric. The observed errors were substantially smaller, approximately 0.25–0.46%, but exact floating-point outputs can vary with reduction order and hardware.

Readiness results are local preconditions for the campaign launcher; repeat them after changing models, code, GPU count, environment or hardware. Historical JSON files shipped under `benchmarks/` are evidence, not substitutes for your local checks.

## 6. Reproduce all twelve renders

```bash
python scripts/run_campaign.py --gpus 4 --attempt replication4_a1
```

The launcher checks for an existing GPU job, holds a local campaign lock, checks readiness records, validates the source revision and hardware profile, and records the command/environment and source hashes. It does not stop other services. It launches one resident engine for:

1. 5-second dialogue: one warmup, then three measurements.
2. 10-second long dialogue: one warmup, then three measurements.
3. 10-second short speech and silence: one warmup, then three measurements.

Follow progress from another terminal:

```bash
tail -f "$H3_RUN_ROOT/replication4_a1.log"
```

Results are written under `H3_RUN_ROOT/replication4_a1/<case>/<warmup|repeat1|repeat2|repeat3>/`. Every completed render has `output.mp4` and `result.json`. A finished campaign also has `summary.json`.

The runner measures until rank zero finishes saving the MP4 and the control barrier completes. It then validates dimensions, decoded frame count and full audio/video decoding outside the timer. Per-rank audits require four DiT forwards and the expected attention profile. The summary tool additionally verifies H.264, 24 FPS, AAC stereo at 32 kHz and each MP4 hash.

To revalidate and print a summary:

```bash
python scripts/summarize.py "$H3_RUN_ROOT/replication4_a1"
```

Choose a new attempt name for a rerun. Existing results are not overwritten. A failed attempt remains available for diagnosis; it is not silently resumed or combined with another attempt.

## 7. Generate a single clip with the upstream CLI

After the same setup, patch, model download and checks, the upstream CLI is also usable:

```bash
export H3_ENCODER_TP=1
python -c 'import json; from pathlib import Path; Path("prompt.txt").write_text(json.load(open("configs/cases.json"))[0]["prompt"])'
torchrun --standalone --nproc_per_node=4 "$SANA_ROOT/models/minimax_h3/Sol-H3/infer.py" \
  --model "$H3_MODEL_ROOT" \
  --adapter "$H3_ADAPTER_PATH" \
  --task t2v --duration 5 --seed 42 \
  --prompt-file prompt.txt --output "$H3_RUN_ROOT/example.mp4" --warmup
```

Keep the other profile environment variables from step 4. This command uses upstream logging and does not reproduce the detailed MP4-inclusive benchmark methodology; use `run_campaign.py` for comparable measurements.

## 8. Assess the result

Verify all requested records completed before reporting medians. Play the original video with audio and check the requested words, repetitions, unexpected speech, silence interval and lip synchronization, as well as visual detail and motion. A low ASR word error rate alone does not establish fidelity.

The native “5s” output contains 124 frames and the “10s” output 243 frames at 24 FPS; those names are model presets, not promises of exactly 5.000/10.000 seconds. Preserve that output for the benchmark. Trimming to 120/240 frames changes the comparison.

Warm results exclude loading and the per-case warmup. Original engine startup was approximately 199.9 seconds; disk speed and compiler-cache state can change startup substantially. The warm timings include prompt encoding on every request; no prompt-embedding cache was used.

