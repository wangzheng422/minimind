# MiniMind Colab L4 profile

| Field | Value |
|---|---|
| Round | 6 |
| Status | Implemented and verified |
| Target | NVIDIA L4, 22 GB-class GPU |

## Recommendation

Run the micro learning route now. It is intentionally small and fits comfortably on the available L4. Leave `RUN_ZERO_PROFILE = False` until micro pretrain, SFT, and inference all complete.

When ready to try Zero, set `RUN_ZERO_PROFILE = True` and retain `ZERO_BATCH_SIZE = 8` initially. Monitor the one-step output's peak GPU-memory field before increasing batch size. This changes the effective batch from the repository's default Zero profile, so it is an L4-safe learning variant rather than an exact reference reproduction. The A100 assertion is optional and remains available by setting `REQUIRE_A100 = True`.

## Delivered adjustment

| Artifact | Change | Exact final copy |
|---|---|---|
| [colab/minimind_colab_learning.ipynb](../colab/minimind_colab_learning.ipynb) | L4-compatible preflight and conservative Zero batch | [archived notebook](files/wzh-solution-2026-07-12-20-45/minimind_colab_learning.ipynb) |
| [colab/README.md](../colab/README.md) | Explain L4 operating guidance | [archived README](files/wzh-solution-2026-07-12-20-45/README.md) |
| [tests/test_minimind_colab.py](../tests/test_minimind_colab.py) | Lock accelerator-profile behavior | [archived test](files/wzh-solution-2026-07-12-20-45/test_minimind_colab.py) |
| [colab/minimind_colab.py](../colab/minimind_colab.py) | BF16 guidance explicitly accepts L4 and A100 | [archived runner](files/wzh-solution-2026-07-12-20-45/minimind_colab.py) |

Full evidence is in [round 6 steps](../wzh-steps/wzh-steps-2026.07.12.20.45.md). Five regression tests and the final whitespace check pass.
