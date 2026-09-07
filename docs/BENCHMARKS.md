# Benchmark methodology and measured evidence

All measurements below are from the completed four-GPU `tp4_a4` campaign on September 7, 2026. The campaign produced 12 successful renders: one warmup and three measured requests for each of three cases. The complete exported records are in [measurements.json](../benchmarks/2026-09-07/measurements.json); numerical gates are in [numerical_checks.json](../benchmarks/2026-09-07/numerical_checks.json).

## Protocol

One resident Sol-H3 engine served all cases. Each duration/prompt received its own full warmup. Startup took 199.900979 seconds and is excluded from warm timings. Compiler caches from earlier probes may have been present; the reported warmups are not claimed to be pristine-cache compilation measurements.

The profile uses the original BF16 base model, FastH3 dense-datafree adapter, SOL/BSA attention, encoder TP4, DiT Ulysses4, 1344×768, 24 FPS, seed 42 and native audio. Five scheduler points execute four actual DiT forwards on every rank. Video/audio scheduler shifts are 12/3. The exact prompts are in [configs/cases.json](../configs/cases.json).

**Full request** is wall-clock time from invoking generation through completed MP4 saving and the post-save Gloo control barrier. **Pipeline** is the engine timing including text encoding, denoising, video/audio decode and synchronization, excluding MP4 saving. Stage timers synchronize CUDA. FFprobe, full AV decode validation and hashing occur after the request timer stops. All prompts are encoded on every request; there is no prompt-embedding cache.

## Every repetition

| Case | Warmup, s | Repeat 1, s | Repeat 2, s | Repeat 3, s | Full-request median, s | Pipeline median, s | Distinct measured MP4 hashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-second dialogue | 19.190289 | 12.123239 | 12.189882 | 12.209885 | 12.189882 | 11.595089 | 3 |
| 10-second long dialogue | 55.368814 | 28.251288 | 28.226655 | 28.162317 | 28.226655 | 27.080013 | 2 |
| 10-second short speech | 28.761406 | 28.210932 | 28.199411 | 28.174237 | 28.199411 | 27.085745 | 3 |

Fixed seed did not make all output files byte-identical. The distinct-hash counts are reported rather than hidden; they do not by themselves identify whether differences originate in model arithmetic or encoding. No bitwise determinism claim is made.

## Stage medians

| Case | Text encoding, s | Denoise, s | Video decode, s | Audio decode, s |
|---|---:|---:|---:|---:|
| 5-second dialogue | 0.046463 | 9.799273 | 1.687695 | 0.036004 |
| 10-second long dialogue | 0.057737 | 23.555695 | 3.362212 | 0.063326 |
| 10-second short speech | 0.058539 | 23.560996 | 3.364222 | 0.062907 |

Stage medians need not add up exactly to the total median. MP4 saving and other overhead are outside these four stage rows. The eight-GPU forecast uses a residual explicitly defined from these medians rather than claiming another directly measured stage.

## Historical comparisons

| Case | Local 20-step control, s | Previous vLLM4, s | New Sol-H3, s | 20-step / Sol-H3 | vLLM4 / Sol-H3 |
|---|---:|---:|---:|---:|---:|
| 5-second dialogue | 178.083 | 35.987586 | 12.189882 | 14.61× | 2.95× |
| 10-second long dialogue | 550.181 | 87.826154 | 28.226655 | 19.49× | 3.11× |
| 10-second short speech | N/A | 87.926227 | 28.199411 | N/A | 3.12× |

The local ComfyUI baseline used 20 steps with an INT8/mixed-precision configuration on an earlier host. It is not an independently measured production workflow. The previous four-GPU vLLM-Omni runs used BF16 LightX Turbo8, with three repetitions for 5s, two for the long 10s case and one for short speech. The prompts/seed are matched, but the runtime, adapter, attention/precision profiles, host and frame counts differ. Ratios therefore describe these configurations jointly; they do not isolate a runtime-only or four-GPU-only improvement.

Sol-H3 preserves native 124/243 frames, while the historical vLLM references use 120/240 frames. Output includes H.264 video and native AAC stereo audio at 32 kHz. The model presets are approximately 5/10 seconds; the frame counts divided by 24 are about 5.17/10.13 seconds.

## Validation results

- All 12 render records completed with four audited ranks and four DiT forwards per rank.
- Each request used the expected SOL/BSA configuration and 144 sparse attention calls per rank, with only the intended first-step/first-two-layer dense policy.
- INT8 QKV transport was active on all 200 attention calls; attention output transport was FP8.
- Resolution, 24 FPS, native frame count, H.264 video, AAC stereo/32 kHz audio, complete AV decoding and MP4 hashes passed.
- The original public review copies of three representative clips were byte-identical to repeat 1; the repository retains their hashes in the records, without depending on a temporary hosting URL.

The full-checkpoint encoder parity errors on rank zero were 0.0046477313, 0.0025019471 and 0.0027386916 relative L2. All four ranks passed a preselected 0.02 threshold with finite outputs. Maximum absolute differences were 64, 8 and 8, respectively; large activation magnitudes make relative and absolute measures different. Tiny FP32 errors were below 2.9e-7 and tiny BF16 errors below 0.007. These are implementation checks, not evidence that a four-step clip matches the quality of a 20-step reference.

## Failed attempts and exclusions

| Attempt | Outcome | Treatment |
|---|---|---|
| a1 | Component metadata still targeted remote Hub locations despite the outer local model path | Failed before generation; motivated the local component-loading fix |
| a2 | Four ranks failed CUDA OOM at pipeline transfer to GPU | No completed generation or usable latency; motivated encoder TP |
| tp4_a3 | Valid 124-frame warmup MP4 was saved, then post-save coordination stalled | Diagnostic clip preserved in the original workspace; excluded from benchmark statistics |
| tp4_a4 | All three cases, each with one warmup and three measurements, completed | Sole source of the reported Sol-H3 medians |

The successful runner separated CPU control synchronization from GPU math with Gloo and used decoded frame counting for fragmented MP4. Original failed-attempt artifacts remain preserved in the source workspace; this repository does not publish unrelated service logs or private handoff files.

## Quality and claim boundaries

Human review of visual detail, temporal stability, exact speech, unwanted additional words, the requested silence interval and lip synchronization is pending. ASR WER=0 would not be sufficient acceptance. The original 10× goal with preserved quality remains unachieved.

The measured historical latency ratios exceed 5–10× against the available local 20-step controls, but the ratio against the previous four-GPU vLLM configuration is approximately 3×. There is no unpatched successful RTX control, no same-host one/four/eight-GPU ablation and no eight-GPU RTX generation in this campaign. The [eight-GPU document](EIGHT_GPU.md) labels its calculations as forecasts.
