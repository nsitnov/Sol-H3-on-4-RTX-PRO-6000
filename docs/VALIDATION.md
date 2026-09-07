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

The first warmup has a null upstream previous-request sparse counter. Validation of the portable summary caught that distinction; the final checker uses cumulative sparse/dense counters and completed request count, covering every warmup as well as measured requests.

A complete fresh-host dependency installation and a second 12-render generation campaign with the portable harness were not performed. Full-checkpoint encoder parity and full video/audio generation are supported by the original campaign records. Eight-GPU generation and eight-rank numerical checks have not been performed on RTX PRO 6000.

These boundaries matter: patch application, syntax checks and small numerical tests are useful replication checks, but they do not establish fresh-host end-to-end timing or output quality.

