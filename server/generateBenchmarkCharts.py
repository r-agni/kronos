#!/usr/bin/env python3
"""Charts for the measured RoboReward sample and FLORA-paper baselines.

KRONOS bars are drawn only from runs/benchmarks/honest_roboreward_eval.json.
Unmeasured KRONOS series are omitted. Competitor task scores are the FLORA
paper transcription, not a re-measurement. Kimi and Sol dollar costs are not
plotted: they were not found in a source opened for this run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "charts"
PUBLIC = ROOT / "public" / "charts"
EVAL_PATH = ROOT / "runs" / "benchmarks" / "honest_roboreward_eval.json"

plt.rcParams.update({
    "figure.facecolor": "#0b1017",
    "axes.facecolor": "#0f1722",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#f8fafc",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#1e293b",
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
})

# FLORA paper transcription. Claude cost and plotted quality use the corrected
# figures ($41.70, 30.4%). Not re-measured in this repository.
PAPER_MODELS = ["Claude Opus 5", "Kimi K3", "GPT-5.6 Sol", "GPT-6 Astra", "NVIDIA FLORA"]
PAPER_COLORS = ["#64748b", "#94a3b8", "#f59e0b", "#a855f7", "#22c55e"]
PAPER_QUALITY = [30.4, 43.2, 47.9, 53.7, 57.1]
# Index-aligned with PAPER_MODELS. None means the dollar figure was not found.
PAPER_COST = [41.70, None, None, 88.49, 0.72]
WGO_SEG = [25.5, 26.1, 33.7, 46.0, 50.3]
WGO_CAP = [49.3, 50.6, 56.6, 77.4, 64.7]
WGO_JOINT = [15.4, 16.7, 24.6, 39.0, 34.5]
ROBOT_SEG = [45.9, 41.2, 45.8, 71.9, 66.9]
ROBOT_CAP = [49.7, 48.0, 69.6, 80.2, 73.7]
ROBOT_JOINT = [23.6, 22.2, 33.6, 57.9, 51.0]
PAPER_MAE = [88.2, 68.2, 78.0, 55.4, 40.0]
PAPER_TAU = [63.1, 63.5, 64.2, 82.3, 72.0]
PAPER_RUBRIC = [17.1, 15.6, 8.4, 3.6, 6.5]
PAPER_MACRO = [33.9, 31.2, 36.3, 40.7, 46.2]
PAPER_RECALL = [0.3, 0.3, 1.0, 0.0, 53.3]
PAPER_FALSE_ALARM = [1.8, 2.1, 2.4, 0.9, 11.0]


def save_fig(fig, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for folder in (DOCS, PUBLIC):
        fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def measured_levels() -> dict:
    doc = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if doc.get("status") != "complete":
        raise SystemExit(f"{EVAL_PATH} is not a complete eval")
    out = {}
    for level in ("high", "low"):
        abs_errors = []
        exact = 0
        failed = 0
        attempted = 0
        for ep in doc["episodes"]:
            call = (ep.get("calls") or {}).get(level)
            if call is None:
                continue
            attempted += 1
            pred = call.get("pred")
            if pred is None:
                failed += 1
                continue
            err = abs(int(pred) - int(ep["gold"]))
            abs_errors.append(err)
            exact += int(err == 0)
        n = len(abs_errors)
        if n == 0:
            raise SystemExit(f"no scored episodes for thinking {level}")
        block = {
            "n_attempted": attempted,
            "n_scored": n,
            "n_failed": failed,
            "sum_abs_error": int(sum(abs_errors)),
            "exact_match_count": exact,
            "mae_x100": 100.0 * sum(abs_errors) / n,
            "exact_pct": 100.0 * exact / n,
        }
        summary = doc["summary"][level]
        if summary["sum_abs_error"] != block["sum_abs_error"] or summary["n_scored"] != n:
            raise SystemExit(f"summary for {level} does not match episode arithmetic")
        out[level] = block
    protocol = doc["protocol"]
    if protocol.get("seed") != 0 or protocol.get("sample_n") != 30:
        raise SystemExit("eval protocol is not the pre-registered seed-0 sample of 30")
    return out


def plot_fig1_pareto():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xscale("log")
    for cost, quality, name, color in zip(PAPER_COST, PAPER_QUALITY, PAPER_MODELS, PAPER_COLORS):
        if cost is None:
            continue
        ax.scatter(cost, quality, s=220, color=color, edgecolors="#ffffff", linewidths=1.2, zorder=4)
        ax.annotate(
            f"{name}\n({quality:.1f}%, ${cost:.2f}/hr)",
            (cost * 1.08, quality + 1.1),
            fontsize=8.5,
            color=color,
        )
    ax.set_xlim(0.4, 160)
    ax.set_ylim(24, 68)
    ax.set_xlabel("Cost per video hour from the FLORA paper ($/hr, log scale)")
    ax.set_ylabel("Mean quality in the FLORA paper (% ↑)")
    ax.set_title(
        "Figure 1: FLORA paper quality vs cost\nKimi and Sol $/hr not plotted (unknown). KRONOS cost and mean quality: not measured.",
        fontsize=11,
        fontweight="bold",
    )
    save_fig(fig, "fig1_pareto_cost_quality")


def plot_fig2_captioning_segmentation():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(PAPER_MODELS))
    width = 0.25
    ax = axes[0]
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.bar(x - width, WGO_SEG, width, label="Seg F1@0.75 ↑", color="#38bdf8")
    ax.bar(x, WGO_CAP, width, label="Gold-Window Caption Acc ↑", color="#a855f7")
    ax.bar(x + width, WGO_JOINT, width, label="Joint Semantic F1@0.75 ↑", color="#00f0ff")
    ax.set_xticks(x)
    ax.set_xticklabels(PAPER_MODELS, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (% ↑)")
    ax.set_title("A. WGO-Bench (FLORA paper)", fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", framealpha=0.3, fontsize=8)

    ax2 = axes[1]
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax2.bar(x - width, ROBOT_SEG, width, label="Seg F1@0.50 ↑", color="#38bdf8")
    ax2.bar(x, ROBOT_CAP, width, label="Caption Point Quality ↑", color="#a855f7")
    ax2.bar(x + width, ROBOT_JOINT, width, label="Joint Point F1@0.50 ↑", color="#22c55e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(PAPER_MODELS, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Score (% ↑)")
    ax2.set_title("B. Robot200 (FLORA paper)", fontweight="bold", fontsize=10.5)
    ax2.legend(loc="upper left", framealpha=0.3, fontsize=8)
    fig.suptitle(
        "Figure 2: Paper captioning and segmentation. KRONOS was not measured on these tasks.",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, "fig2_captioning_and_segmentation")


def plot_fig3_reward_benchmarks():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    x = np.arange(len(PAPER_MODELS))
    series = [
        (PAPER_MAE, "A. RoboReward MAE×100 ↓\nFLORA paper, 2,831 episodes", 110),
        (PAPER_TAU, "B. Robometer OOD τ ↑\nFLORA paper", 100),
        (PAPER_RUBRIC, "C. Rubric L1 ↓\nFLORA paper", 20),
    ]
    for ax, (vals, title, ylim) in zip(axes, series):
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        bars = ax.bar(x, vals, color=PAPER_COLORS, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(PAPER_MODELS, rotation=22, ha="right", fontsize=8)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontweight="bold", fontsize=9.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + ylim * 0.02, f"{h:.1f}", ha="center", fontsize=8)
    fig.suptitle(
        "Figure 3: FLORA paper reward tables. KRONOS sample MAE is Figure 5, not these bars.",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    save_fig(fig, "fig3_outcome_robometer_rubric")


def plot_fig4_dense_process_rewards():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(PAPER_MODELS))
    width = 0.25
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.bar(x - width, PAPER_MACRO, width, label="3-Class Macro-F1 (% ↑)", color="#38bdf8")
    ax.bar(x, PAPER_RECALL, width, label="0.5s Failure Recall (% ↑)", color="#22c55e")
    ax.bar(x + width, PAPER_FALSE_ALARM, width, label="Normal-step false alarms (% ↓)", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(PAPER_MODELS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        "Figure 4: DenseRewardBench from the FLORA paper\nKRONOS failure recall, false alarms, and macro-F1: not measured",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(loc="upper left", framealpha=0.35)
    save_fig(fig, "fig4_dense_process_rewards")


def plot_fig5_measured_sample(levels: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.axis("off")
    ax.set_title("A. DPO loss and survival", fontweight="bold", fontsize=11)
    ax.text(
        0.02,
        0.55,
        "Not measured.\n\nThe published loss curve and the 98.2% survival\nfigure were not recomputed from a label-free run.\nThey are not KRONOS results.",
        transform=ax.transAxes,
        fontsize=11,
        va="center",
        color="#e2e8f0",
    )

    high = levels["high"]
    low = levels["low"]
    labels = [
        f"thinking low\n{low['exact_match_count']}/{low['n_scored']} exact",
        f"thinking high\n{high['exact_match_count']}/{high['n_scored']} exact",
    ]
    mae_vals = [low["mae_x100"], high["mae_x100"]]
    exact_vals = [low["exact_pct"], high["exact_pct"]]
    x = np.arange(len(labels))
    w = 0.34
    ax2 = axes[1]
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    b_mae = ax2.bar(x - w / 2, mae_vals, w, label="MAE×100 ↓", color="#f59e0b")
    b_ex = ax2.bar(x + w / 2, exact_vals, w, label="Exact match % ↑", color="#22c55e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.set_title(
        "B. gemini-4-argon on a seed-0 sample\n"
        f"n_scored={high['n_scored']}, n_failed={high['n_failed']} of {high['n_attempted']} "
        "(not the 2,831-episode MAE)",
        fontweight="bold",
        fontsize=10,
    )
    ax2.legend(loc="upper right", framealpha=0.3, fontsize=8)
    for b in b_mae:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.2f}", ha="center", fontsize=8, color="#fbbf24")
    for b in b_ex:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.2f}%", ha="center", fontsize=8, color="#4ade80")
    fig.suptitle(
        f"Figure 5: Measured sample. Low MAE×100 = 100×({low['sum_abs_error']}/{low['n_scored']}). "
        f"High MAE×100 = 100×({high['sum_abs_error']}/{high['n_scored']}).",
        fontsize=11,
        fontweight="bold",
        y=1.04,
    )
    save_fig(fig, "fig5_gpu_dpo_and_thinking_ablation")


def main() -> None:
    levels = measured_levels()
    plot_fig1_pareto()
    plot_fig2_captioning_segmentation()
    plot_fig3_reward_benchmarks()
    plot_fig4_dense_process_rewards()
    plot_fig5_measured_sample(levels)
    # save_fig already writes both folders. Copy again in case a figure was saved once.
    for folder_src, folder_dst in ((DOCS, PUBLIC),):
        for path in folder_src.glob("fig*.*"):
            shutil.copy2(path, folder_dst / path.name)
    print("GENERATED_5_BENCHMARK_FIGURES_OK")
    print(json.dumps(levels, indent=2))


if __name__ == "__main__":
    main()
