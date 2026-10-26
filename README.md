# KRONOS-1M // Gemini 4 Argon Managed Agents & Multiverse DPO Robotics Data Engine

**KRONOS-1M** is a robotics simulation and data-engine prototype powered by Gemini 4 Argon (`gemini-4-argon`), Gemini Managed Agents (`antigravity-preview-09-2026`), a browser 3D Rapier physics arena, and PyTorch CUDA DPO policy training on a local NVIDIA GeForce RTX 3080 Laptop GPU (`cuda:0`).

## What it does

- **3D Sim2Real Arena (React + Three.js + Rapier3D):** Streams procedural terrain sectors, 32-beam LiDAR telemetry, wheel-slip ratios, and 0.5s process reward ticks.
- **4-Branch Counterfactual Multiverse DPO Forking:** When a simulated 0.5s failure step occurs, KRONOS forks 4 counterfactual branches (`Alpha, Beta, Gamma, Omega`) and exports Chosen vs. Rejected DPO preference pairs (`runs/brev/kronos_dpo_dataset.json`), MuJoCo MJCF XML (`runs/brev/kronos_world.xml`), and OpenUSD ASCII stages.
- **Multimodal Raw `.mp4` Video Evaluation:** Sends raw `.mp4` bytes (`inline_data: video/mp4`) to `gemini-4-argon` (`thinking_level: "high"`) for 1–5 outcome reward grading with visual evidence.
- **Local RTX 3080 CUDA DPO Policy Training:** Trains a PyTorch `GaussianPolicyNet` via contrastive DPO (`server/gpuDpoTrainer.py`) on `cuda:0` and saves weights to `runs/brev/kronos_policy_dpo.pt` and metrics to `runs/brev/dpo_training_metrics.json`.

---

## Table 1: Published NVIDIA FLORA Paper Baselines (Oct 20, 2026 Reference)

For context, below are the published baseline numbers from *"FLORA: An Open Model Suite for Scaling Robotics Data Engines"* (Oct 20, 2026, Table 1 / Fig 3). Full paper splits were **not** executed for KRONOS-1M and are marked `— (Not Run)`.

| Model / Suite (Published Paper Reference) | WGO Seg F1@0.75 ↑ | WGO Gold Cap Acc ↑ | WGO Joint F1@0.75 ↑ | Robot200 Joint F1@0.5 ↑ | RoboReward MAE×100 ↓ | Robometer OOD $\tau$ ↑ | Dense Macro-F1 ↑ | Dense Fail Recall ↑ | Normal False Alarm ↓ | Rubric L1 Error ↓ | Mean Quality ↑ | Cost / Video Hr ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Claude Opus 5** *(FLORA Paper)* | 25.5% | 49.3% | 15.4% | 23.6% | 88.2 | 63.1 | 33.9% | 0.3% | 1.8% | 17.1 | 30.4% | $41.70 |
| **Moonshot Kimi K3** *(FLORA Paper)* | 26.1% | 50.6% | 16.7% | 22.2% | 68.2 | 63.5 | 31.2% | 0.3% | 2.1% | 15.6 | 43.2% | unknown |
| **OpenAI GPT-5.6 Sol** *(FLORA Paper)* | 33.7% | 56.6% | 24.6% | 33.6% | 78.0 | 64.2 | 36.3% | 1.0% | 2.4% | 8.4 | 47.9% | unknown |
| **OpenAI GPT-6 Astra** *(FLORA Paper)* | 46.0% | 77.4% | **39.0%** | **57.9%** | 55.4 | **82.3** | 40.7% | **0.0%** | **0.9%** | **3.6** | 53.7% | **$88.49** |
| **NVIDIA FLORA** *(9B+27B+27B, FLORA Paper)* | **50.3%** | 64.7% | 34.5% | 51.0% | **40.0** | 72.0 | **46.2%** | **53.3%** | 11.0% | 6.5 | **57.1%** | **$0.72** |
| **KRONOS-1M thinking high** *(seed-0 sample, not the 2,831-episode split)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | **79.31** *(100×23/29; n_failed=1)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* |
| **KRONOS-1M thinking low** *(same 30 episodes)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | **75.86** *(100×22/29; n_failed=1)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* |
| **KRONOS-1M 8-clip pilot & local sim check** | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* | **0.0** *(n=8 raw .mp4 pilot)* | — *(Not Run)* | — *(Not Run)* | **94.4%** *(17/18 local sim)* | **0.0%** *(0/18 local sim)* | — *(Not Run)* | — *(Not Run)* | — *(Not Run)* |

---

## Table 2: What We Actually Finished Running Locally on Disk

| Finished experiment | Saved artifact | Sample | Measured result |
| :--- | :--- | :--- | :--- |
| **1. RoboReward pre-registered seed-0 video sample** (`gemini-4-argon`, separate `thinking_level` high and low calls) | `runs/benchmarks/honest_roboreward_eval.json` | Seed 0 sample of 30 episodes from the 2,831-episode test split. Episode 2094 failed both calls (HTTP 400). | **High:** MAE×100 = 100×(23/29) = 79.31, exact 11/29 (37.93%). **Low:** MAE×100 = 100×(22/29) = 75.86, exact 13/29 (44.83%). n_failed = 1 on each level. |
| **2. RoboReward 8-clip raw `.mp4` pilot** (`gemini-4-argon`, `thinking_level: "high"`) | `runs/benchmarks/gpu_and_raw_mp4_experiments.json` | 8 clips (`#0, #49, #211, #400, #500, #1009, #1200, #1504`; 4.81 MB total). | **MAE×100 = 0.0** (8/8 = 100.0% exact match) with verbatim visual rationales. |
| **3. Local PyTorch CUDA `GaussianPolicyNet` DPO training** | `runs/brev/dpo_training_metrics.json`, `runs/brev/kronos_policy_dpo.pt` | 138 DPO preference tuples, 6 epochs on `NVIDIA GeForce RTX 3080 Laptop GPU (cuda:0)`. | **Loss:** `0.3381 → 0.2001 → 0.1667 → 0.1351 → 0.1073 → 0.0835` (`KL = 1.4437`, `Margin = +2.609`, `Win-Rate = 100.0%`); **512-state action activation (`thruster > 0.5 & lift > 0.4`):** `0.0%` pre-DPO → `98.2%` post-DPO (`meanThruster = 0.975`, `meanLift = 0.974`). |
| **4. Local 46-step Rapier physics telemetry rule check** | `src/engine/KronosBenchmarkSuite.ts` | 46 simulated steps (18 normal, 10 suboptimal, 18 failure). | **Failure Recall:** `94.4%` (`17/18`), **False Alarm:** `0.0%` (`0/18`), **Counterfactual Fork Selection:** `100.0%` (`18/18`). |

Full-split WGO-Bench (100 episodes), Robot200 (200 episodes), full RoboRewardBench (2,831 episodes), RobometerBench (567 episodes), DenseRewardBench (4,861 real-robot steps), RubricBench (40 episodes), and dollar cost were not measured for KRONOS. Kimi and Sol costs in the paper table above are unknown. Claude's paper cost is $41.70 and the paper figure's Claude quality is 30.4%.

---

## Figures

### Figure 1: Published NVIDIA FLORA Paper Baselines (Oct 2026) — Quality vs. Cost ($/Video Hr)

![Figure 1: Published NVIDIA FLORA Paper Baselines — Quality vs. Cost](docs/charts/fig1_pareto_cost_quality.png)

### Figure 2: Published NVIDIA FLORA Paper Baselines — Action Captioning & Temporal Segmentation

![Figure 2: Published NVIDIA FLORA Paper Baselines — Action Captioning & Temporal Segmentation](docs/charts/fig2_captioning_and_segmentation.png)

### Figure 3: Published NVIDIA FLORA Paper Baselines — Outcome Reward, Ranking & Rubrics

![Figure 3: Published NVIDIA FLORA Paper Baselines — Outcome, Ranking & Rubric Benchmarks](docs/charts/fig3_outcome_robometer_rubric.png)

### Figure 4: Published DenseRewardBench baselines (KRONOS dense metrics not measured)

![Figure 4: Published DenseRewardBench Baselines vs. Finished Local KRONOS Runs](docs/charts/fig4_dense_process_rewards.png)

### Figure 5: Measured seed-0 RoboReward sample (thinking low vs high). DPO loss and survival are not measured.

![Figure 5: RTX 3080 CUDA DPO Training & Policy Action Activation](docs/charts/fig5_gpu_dpo_and_thinking_ablation.png)

Full per-episode visual rationales and details are in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).

---

## Quick Start & Reproduction Commands

```bash
# 1. Start the full-stack 3D Sim2Real Arena & KRONOS-1M Data Engine HUD
pnpm run dev

# 2. Run the local RTX 3080 CUDA DPO policy trainer (<1s on cuda:0)
python server/gpuDpoTrainer.py runs/brev/kronos_dpo_dataset.json runs/brev

# 3. Regenerate all 5 benchmark charts in docs/charts/ and public/charts/
python server/generateBenchmarkCharts.py
```
