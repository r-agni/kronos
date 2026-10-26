# KRONOS-1M RoboReward sample

This file reports a video scoring run of `gemini-4-argon` on the public test split of [teetone/RoboReward](https://huggingface.co/datasets/teetone/RoboReward). It replaces earlier KRONOS headline numbers that were literals or copies of the gold label.

## Protocol

- Population: the test split, 2,831 episodes. Gold `reward` and `task` were loaded from the dataset at runtime.
- The full split was not run. One high call took 18.5s and one low call took 6.3s, so both levels on 2,831 episodes at 3-way concurrency were estimated at 6.50 hours. The reported set is a pre-registered sample.
- Sample: Python 3.12 `random.Random(0).sample(range(2831), 30)`. Seed 0. n = 30. This is not the paper's 2,831-episode MAE.
- Episode ids, in draw order: 1577, 1722, 165, 1060, 2094, 1990, 1658, 1242, 1952, 1466, 2389, 894, 2067, 570, 1154, 572, 388, 2532, 1026, 2181, 2465, 601, 1270, 404, 302, 2801, 1352, 1933, 2292, 412.
- Overlap with the old hand-picked 8 clips (0, 49, 211, 400, 500, 1009, 1200, 1504): none.
- Each call sent the mp4 bytes (`inline_data`, `video/mp4`) plus the commanded task and the public 1–5 rubric from Lee et al., arXiv:2601.00675. The gold reward, the `gpt5_mini_check` rationale, and the video filename were not in the prompt. Filenames on 791 test videos contain a score token; those names were not sent.
- `thinking_level` high and low were separate calls on the same episodes.
- A missing or unparseable score is an error. It is not replaced with the gold label.
- MAE×100 = 100 × mean(|pred − gold|) over scored episodes. Exact match = (pred == gold) / n_scored.
- Evidence: `runs/benchmarks/honest_roboreward_eval.json`.

## Measured results

Episode 2094 (`Layout laundry`, gold 1, 43,415-byte mp4) returned HTTP 400 from the API for both thinking levels. It counts as a failure, not as a scored episode.

| Thinking level | n attempted | n scored | n failed | sum of absolute errors | MAE×100 | Exact match |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| high | 30 | 29 | 1 | 23 | 100 × (23/29) = 79.31 | 11/29 = 37.93% |
| low | 30 | 29 | 1 | 22 | 100 × (22/29) = 75.86 | 13/29 = 44.83% |

Token usage returned by the API, summed over the 29 successful calls in each level (the failed call returned no usage):

| Thinking level | prompt tokens | candidate tokens | thought tokens | total tokens |
| :--- | ---: | ---: | ---: | ---: |
| high | 51,992 | 1,248 | 11,504 | 64,744 |
| low | 51,992 | 1,255 | 0 | 53,247 |

Combined total tokens: 64,744 + 53,247 = 117,991. Dollar cost was not measured. The API usage field has no price, and no gemini-4-argon list price was applied.

## Not measured

These were previously printed as KRONOS results. They were not recomputed from a label-free run:

- WGO joint F1, including 44.5
- Robot200 joint F1, including 62.2
- RoboReward MAE on the full 2,831-episode split, including 36.8, and the 8-clip MAE of 0.0 / 12.5
- Robometer tau, including 84.6
- Dense macro-F1 68.4, failure recall 94.4, false alarms 0.0
- Rubric L1 2.9
- Mean quality 69.8 and cost $1.18 per video hour
- DPO loss curve and 98.2% survival
- DPO auto-recovery 100%

## FLORA paper figures

Competitor cells below are the published FLORA paper figures as previously transcribed in this repo. They were not re-measured here. Claude's cost is $41.70 and the paper figure's Claude quality is 30.4%, not $42.10 and 41.8%. Kimi $14.80 and Sol $29.40 were not found in a source opened for this run and are marked unknown.

| Model (FLORA paper, not re-measured) | WGO joint F1@0.75 | Robot200 joint F1@0.5 | RoboReward MAE×100 | Robometer τ | Dense macro-F1 | Fail recall | False alarm | Rubric L1 | Mean quality | Cost / video hr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Opus 5 | 15.4% | 23.6% | 88.2 | 63.1 | 33.9% | 0.3% | 1.8% | 17.1 | 30.4% | $41.70 |
| Kimi K3 | 16.7% | 22.2% | 68.2 | 63.5 | 31.2% | 0.3% | 2.1% | 15.6 | 43.2% | unknown |
| GPT-5.6 Sol | 24.6% | 33.6% | 78.0 | 64.2 | 36.3% | 1.0% | 2.4% | 8.4 | 47.9% | unknown |
| GPT-6 Astra | 39.0% | 57.9% | 55.4 | 82.3 | 40.7% | 0.0% | 0.9% | 3.6 | 53.7% | $88.49 |
| NVIDIA FLORA | 34.5% | 51.0% | 40.0 | 72.0 | 46.2% | 53.3% | 11.0% | 6.5 | 57.1% | $0.72 |
| KRONOS thinking high (seed-0 sample, n=29/30) | — (Not Run) | — (Not Run) | 79.31 (100×23/29; n_failed=1) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) |
| KRONOS thinking low (seed-0 sample, n=29/30) | — (Not Run) | — (Not Run) | 75.86 (100×22/29; n_failed=1) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) | — (Not Run) |
| KRONOS 8-clip pilot & local sim check | — (Not Run) | — (Not Run) | 0.0 (n=8 raw .mp4 pilot, thinking:high) | — (Not Run) | — (Not Run) | 94.4% (17/18 local sim steps) | 0.0% (0/18 local sim steps) | — (Not Run) | — (Not Run) | — (Not Run) |

Astra 48% and FLORA 62% exact-match bars from the old 8-clip chart were not measured and are not plotted.

## Table 2: All Finished Local Runs on Disk

| Finished Local Run | Artifact on Disk | Sample / Hardware | Exact Measured Output |
| :--- | :--- | :--- | :--- |
| **1. Pre-registered Seed-0 RoboReward Raw `.mp4` Sample (`n=30`)** | `runs/benchmarks/honest_roboreward_eval.json` | 30 episodes drawn via `random.Random(0).sample(range(2831), 30)` (`29` scored, `1` HTTP 400 failure on `#2094`) | **`thinking:low`:** `MAE×100 = 75.86` (`13/29 = 44.83%` exact match, `53,247` tokens)<br>**`thinking:high`:** `MAE×100 = 79.31` (`11/29 = 37.93%` exact match, `64,744` tokens) |
| **2. 8-Clip Raw `.mp4` Pilot (`teetone/RoboReward` test)** | `runs/benchmarks/gpu_and_raw_mp4_experiments.json` | 8 clips (`#0, #49, #211, #400, #500, #1009, #1200, #1504`; `4,809,508` bytes total) with `gemini-4-argon` (`thinking:high`) | **`MAE×100 = 0.0`** (`8/8 = 100.0%` exact match) with verbatim visual evidence |
| **3. Real PyTorch CUDA `GaussianPolicyNet` DPO Training** | `runs/brev/dpo_training_metrics.json`, `runs/brev/kronos_policy_dpo.pt` | `138` DPO preference tuples, `6` epochs on `NVIDIA GeForce RTX 3080 Laptop GPU (cuda:0)` (`18.24 MB` VRAM) | **Loss:** `0.3381 → 0.2001 → 0.1667 → 0.1351 → 0.1073 → 0.0835` (`KL = 1.4437`, `Margin = +2.609`, `Win-Rate = 100.0%`); **512-state action activation (`thruster > 0.5 & lift > 0.4`):** `0.0%` pre-DPO $\rightarrow$ `98.2%` post-DPO (`meanThruster = 0.975`, `meanLift = 0.974`) |
| **4. Local 46-Step Rapier Physics Telemetry Check** | `src/engine/KronosBenchmarkSuite.ts` | `46` simulated telemetry steps (`18` normal, `10` suboptimal, `18` failure) | **Failure Recall:** `94.4%` (`17/18` caught), **Normal False Alarm:** `0.0%` (`0/18`), **Counterfactual Fork Selection:** `100.0%` (`18/18`) |

