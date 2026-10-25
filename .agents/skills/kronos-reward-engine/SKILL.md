---
name: kronos-reward-engine
description: Unifies Temporal Segmentation, Contextual VLA Action Captioning, 0.5s Dense Process Rewards, and Test-Time Rubric Evaluation across Gemini 4 Argon's 1M token window.
---

# KRONOS-1M Reward & Segmentation Skill

## Instructions
1. Ingest the continuous 60Hz rover telemetry stream (`position`, `velocity`, `wheelSlipRatio`, `tippingMarginDeg`, `obstacleClearanceMeters`, `autonomousState`) and forward viewport frames.
2. Emit 0.5-second dense process reward classifications (`normal`, `suboptimal`, `failure`) grounded in physical thresholds and the active operator rubric.
3. Segment the episode into contiguous `[startSec, endSec]` sub-tasks without 512-token window stitching tears.
