# Validation of this replication package

The published four-GPU timing evidence comes from the original complete campaign, not from the packaging checks listed below. The exact runtime patch and original runner hashes are recorded in the reproduction guide and archive notes.

The packaging checks on September 7, 2026 include:

- Fetching the pinned Sana revision into a fresh source directory through `scripts/setup_source.py` and applying the patch successfully.
- Comparing the resulting upstream diff byte-for-byte with the patch used by the completed campaign.
- Checking all portable Python files for valid syntax and all local Markdown file links for existing targets.
- Running the portable hardware preflight, exact NCCL all-to-all/SM120 BSA probe and tiny FP32/BF16 encoder parity checks on all four RTX PRO 6000 GPUs, using the existing pinned environment.
- Running the portable summary validator against all 12 original result records and MP4 files through read-only links in a separate validation directory.
- Checking every exported result against its original, with only the local MP4 filename removed from `ffprobe.format.filename`.
- Recomputing medians and eight-GPU scenario values from the shipped numerical evidence.
- Running `scripts/download_models.py --verify-only` against all 61 existing model files and the adapter, with all recorded size/hash checks passing.
- Cloning the public GitHub repository independently and comparing every published file byte-for-byte with the reviewed package.

The first warmup has a null upstream previous-request sparse counter. Validation of the portable summary caught that distinction; the final checker uses cumulative sparse/dense counters and completed request count, covering every warmup as well as measured requests.

A complete fresh-host dependency installation and a second 12-render generation campaign with the portable harness were not performed. Full-checkpoint encoder parity and full video/audio generation are supported by the original campaign records. Eight-GPU generation and eight-rank numerical checks have not been performed on RTX PRO 6000.

These boundaries matter: patch application, syntax checks and small numerical tests are useful replication checks, but they do not establish fresh-host end-to-end timing or output quality.

## Image-conditioned extension validation

The original T2V evidence above is unchanged. For I2V/Ref2VA, the clean-source image setup fetched the pinned revision and applied the cumulative patch; all three changed runtime files matched the actual generation source byte-for-byte. The portable whole-layer tiny check was bit-exact for FP32 and BF16 on every hidden state and all four GPUs. The portable Ref2VA verifier passed all additional model/config files, adapter and source images. All four pinned image download URLs independently returned the exact generation input bytes.

The portable image summary validated all eight I2V and eight Ref2VA records, media hashes and per-rank profiles. The original full multimodal parity check was bit-exact for every image case on all ranks, and all 16 original renders passed AV checks. The public page, four clips and four images matched local bytes; MP4 range requests returned 206 and private internal routes remained 404. Python compilation checks passed.

See [image package validation](../benchmarks/2026-09-07-images/package_validation.json). A fresh dependency installation and a second full 16-render campaign using the portable harness were not performed. No eight-GPU image test or human quality acceptance is claimed.
