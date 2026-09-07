# Historical runner provenance

`benchmark_sol_h3_20260907.py` is the exact runner that produced the successful `tp4_a4` measurements. Its SHA256 is:

```text
13177ea3306ab4a1da479ea78ce7ab6bb208c52f3d0aa1ecc56d2284fe1d1793
```

It retains the original machine-specific paths and four-rank audit size as an immutable source record. Those paths are not prerequisites for replication. Run `scripts/benchmark.py` through `scripts/run_campaign.py` for a portable reproduction.

The portable version replaces paths, uses the actual world size, rejects existing output directories and explicitly checks attention counters. The full-encoder parity wrapper uses a Gloo control barrier while rank zero creates the reference. The runtime patch and generation profile are the same; the portable harness itself is not retroactively the source of the recorded timings.

The public measurement export removes only the original local MP4 filename from `ffprobe.format.filename`. Prompts, timing values, rank audits, numeric checks and media SHA256 values are preserved. Checkpoints, generated binary media and internal workspace handoff/state files are not part of this repository.


## Image campaign provenance

`benchmark_sol_h3_image_20260907.py` is the exact original I2V/Ref2VA runner, SHA256 `53230ee285455bb75ea0777c3951711afa63e3f8b0c354bad5417bdd2817fa52`. The exact full multimodal check is archived as `validate_sol_encoder_multimodal_20260907.py`. Both retain historical machine paths for provenance. Use `scripts/run_image_campaign.py` and `scripts/validate_encoder_multimodal.py` with the [image reproduction guide](../docs/IMAGE_TESTS.md) for portable execution. The image patch is cumulative and distinct from the unchanged original T2V patch.
