# Upstream projects and model terms

This repository contains a small adaptation patch, replication scripts, documentation and measurement metadata. It does not bundle the full upstream runtime, model weights, vLLM-Omni implementation or third-party kernel source. Downloads are performed from the original repositories at pinned revisions.

| Project | Role | Source |
|---|---|---|
| NVIDIA Sol-H3 / Sana | Runtime, SOL/BSA attention, Ulysses, fused operators, AdaLN, parallel decode/encoding | [Pinned source](https://github.com/NVlabs/Sana/tree/2936c47637380842aaa4a4488fac5006cc542b70/models/minimax_h3/Sol-H3) |
| MiniMaxAI MiniMax-H3 | Base video/audio model and associated encoder/VAEs | [Pinned model](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) |
| FastVideo / FastH3 | Dense-datafree four-step adapter | [Pinned adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/f509e629374cac104e7f62daecce6d1488a3041d) |
| vLLM-Omni | Prior encoder tensor-parallel design informing the adaptation | [Encoder source](https://github.com/vllm-project/vllm-omni/blob/ae70479619b9c80b37f0ce29558a72913bc46056/vllm_omni/diffusion/models/minimax_h3/encoder.py) |
| Hugging Face Diffusers / Transformers | Modular pipeline and Qwen3-VL model implementation | [Diffusers](https://github.com/huggingface/diffusers/tree/abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc), [Transformers](https://github.com/huggingface/transformers) |
| NVIDIA cuDNN Frontend | Kernel backend dependency | [Pinned source](https://github.com/NVIDIA/cudnn-frontend/tree/29106622617bfd9031a53099a6fbbc5e74a474e9) |

Upstream source files retain their original copyright/license notices. Model and adapter access and use are governed by their publishers' terms. The Sol-H3 patch includes contextual upstream lines; it does not replace those notices or grant rights to model weights. Consult the original projects for their license texts and citations.

All optimization claims in this repository distinguish the existing upstream acceleration stack from the local encoder/load-path adaptation. There is no claim of NVIDIA, MiniMax, FastVideo or vLLM endorsement.

