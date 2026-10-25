---
name: counterfactual-dpo-synthesizer
description: Synthesizes 4-branch counterfactual multiverse trajectories (Chosen vs Rejected DPO preference pairs), live robot morphology co-design (MORPH_ROVER), and MuJoCo MJCF exports for NVIDIA Brev GPU post-training.
---

# Counterfactual Multiverse DPO Synthesizer Skill

## Instructions
1. When a `FAILURE` or `SUBOPTIMAL` process reward is detected (e.g., 64K seam cliff halt, chasm rupture, or excessive wheel slip under crosswind), switch `thinking_level` to `"high"`.
2. Use the Managed Agent `code_execution` sandbox to evaluate 4 counterfactual branches from the exact fork state `s_t`:
   - `alpha_fail` (`rejected`): Unmodified baseline trajectory.
   - `beta_swerve` (`rejected`): High-slip reactive LiDAR swerve.
   - `gamma_crawl` (`rejected`): Wide-track heavy crawl (`trackWidthScale: 1.38`).
   - `omega_optimal` (`chosen`): Optimal co-design (`thruster_hopper` leap + `deployable_bridge` span).
3. Emit executable Argon DSL directives (`MORPH_ROVER`, `MUTATE_PHYSICS`, `DEPLOY_COUNTERMEASURE`) and stage the resulting DPO preference tuple for NVIDIA Brev GPU training.
