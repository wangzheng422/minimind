# MiniMind Colab L4 profile - command evidence

| Field | Value |
|---|---|
| Round | 6 |
| Status | Complete; ready to commit |
| Working directory | `/Users/zhengwan/Desktop/dev/minimind` |
| Shell | `zsh` |
| Live Colab evidence | NVIDIA L4, 22.03 GiB VRAM, BF16 supported, 52.96 GiB RAM, 57.42 GiB free disk |

## Conclusion

The notebook no longer requires an A100 by default. The active L4 passes the required CUDA/BF16 capability check when `REQUIRE_A100 = False`. The micro route can run as-is; the optional Zero route now starts with batch size 8 to preserve VRAM headroom on a 22 GB GPU.

## Changes and verification

Configuration now includes:

```python
REQUIRE_A100 = False
PREFLIGHT_ACCELERATOR_ARG = '--require-a100' if REQUIRE_A100 else ''
ZERO_BATCH_SIZE = 8
```

The preflight command uses the optional argument, while both Zero training commands pass `--batch-size {ZERO_BATCH_SIZE}`.

At cwd `/Users/zhengwan/Desktop/dev/minimind`, the following command ran under `zsh` and exited 0:

```zsh
python -m unittest discover -s tests -p 'test_*.py' -v
```

All five tests passed, including the new L4 configuration regression and the BF16 guidance wording check. [Complete stdout and stderr](files/wzh-steps-2026-07-12-20-45/regression-green.txt).

## Exact artifacts

Final notebook, README, and regression test were archived under `wzh-solution/files/wzh-solution-2026-07-12-20-45/`. SHA-256 values and byte counts are available in [artifact-sha256.txt](files/wzh-steps-2026-07-12-20-45/artifact-sha256.txt) and [artifact-bytes.txt](files/wzh-steps-2026-07-12-20-45/artifact-bytes.txt).

Final whitespace check:

```zsh
git diff --check -- . ':(exclude)wzh-steps/files/**'
```

Exit code 0; [complete output](files/wzh-steps-2026-07-12-20-45/diff-check-final.txt).
