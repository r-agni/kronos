# KRONOS-1M // Gemini 4 Argon Managed Agents & Multiverse DPO Robotics Data Engine

**KRONOS-1M** is a closed-loop **Physical Auto-Research & Robotics Data Engine** powered by **Gemini 4 Argon (1M Unified Context)**, **Gemini Managed Agents (`antigravity-preview-09-2026`)**, and **NVIDIA GeForce RTX 3080 CUDA / Brev GPU (`pg-solvers`)** DPO post-training.

It unifies all 5 robotics data-engine primitives (**Temporal Sub-Task Segmentation**, **Contextual VLA Action Captioning**, **Outcome Rewards**, **0.5s Dense Process Rewards**, and **Test-Time Rubric Grading**) with a 6th closed-loop capability: **4-Branch Counterfactual Multiverse DPO Forking & Live Hardware/Physics Co-Design**.

---

## Benchmark Figures: KRONOS-1M vs. NVIDIA FLORA & OpenAI GPT-6 Astra

### Figure 1: Overall Benchmark Quality (%) vs. Annotation Cost ($/Video Hr)

![Figure 1: Overall Quality vs. Cost Pareto Frontier](docs/charts/fig1_pareto_cost_quality.png)

### Figure 2: Timestamped Action Captioning & Temporal Segmentation (`WGO-Bench` & `DenseCaptionBench-Robot200`)

![Figure 2: Timestamped Action Captioning & Temporal Segmentation](docs/charts/fig2_captioning_and_segmentation.png)

### Figure 3: Outcome Reward (`RoboRewardBench`), Trajectory Ranking (`RobometerBench`) & Rubrics (`RubricBench`)

![Figure 3: Outcome Reward, Trajectory Ranking & Test-Time Rubric Benchmarks](docs/charts/fig3_outcome_robometer_rubric.png)

### Figure 4: Dense 0.5s Process Rewards (`DenseRewardBench`) — Failure Recall vs. False Alarms

![Figure 4: DenseRewardBench 0.5s Step Process Rewards](docs/charts/fig4_dense_process_rewards.png)

### Figure 5: RTX 3080 CUDA PyTorch DPO Convergence & Gemini 4 Argon `thinking_level` Ablation

![Figure 5: RTX 3080 CUDA DPO Policy Convergence & Thinking Ablation](docs/charts/fig5_gpu_dpo_and_thinking_ablation.png)

Full per-episode visual evidence and analysis are documented in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).

---

## Quick Start & Reproduction Commands

```bash
# 1. Start the full-stack 3D Sim2Real Arena & KRONOS-1M Data Engine HUD
pnpm run dev

# 2. Run the end-to-end verification suite (Rapier WASM + Managed Agents + RTX 3080 DPO)
pnpm run verify

# 3. Run the live Hugging Face benchmark suite (teetone/RoboReward, WGO-Bench, Robometer)
pnpm run benchmark

# 4. Run parallel RTX 3080 CUDA + Raw .mp4 Video Multimodal Experiments
pnpm run gpu:experiments

# 5. Regenerate all 5 publication benchmark charts in docs/charts/
python server/generateBenchmarkCharts.py
```
