"""Record a 60-second (1-minute) live HD demo video and high-resolution screenshots of the
KRONOS-1M (Ocean + Crusoe style) Autonomous Robotics Data Engine & Multiverse DPO Dashboard.

Walks through all 6 core stages & extra simulation/training features:
1. [00s - 10s] 1 · Run & Label: Live 3D rover simulation, 32-beam LiDAR, 6-stage pipeline,
   0.5s dense process rewards strip, and live Gemini 4 Argon 1M-context token & DSL stream.
2. [10s - 20s] Curveball Hazard & 64K Fractured Baseline vs 1M Unified Context:
   Inject Chasm Rupture, toggle 64K fractured seams, and trigger 4-Branch Counterfactual Multiverse DPO Fork.
3. [20s - 30s] 2 · Data & Captions: Inspect Temporal Sub-Task Segmentation (tIoU >= 0.75),
   Contextual VLA Action Captions, 0.5s Dense Process Rewards table, and live Test-Time Rubric presets.
4. [30s - 40s] 3 · Multiverse DPO & Hardware Co-Design: Inspect Chosen (Ghost Omega, 4.9/5) vs
   3 Rejected counterfactual branches, apply Thruster Hopper morphology co-design & Carbon Bridge in 3D.
5. [40s - 50s] 4 · Video Checks & 5 · GPU Train & Sim: Inspect real .MP4 multimodal evaluations
   (Gold vs Astra vs FLORA vs Kronos) and run Contrastive DPO Training on NVIDIA Brev / RTX 3080 CUDA
   (0.0% -> 98.2% post-DPO survival across 278,528 GPU steps).
6. [50s - 60s] 6 · FLORA vs Astra Benchmark & Crusoe Tactical Theme Toggle:
   Full paper-protocol benchmark table, open Benchmark Modal, and return to live 3D simulation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

_GYAN = (
    Path.home()
    / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
)
FFMPEG = os.environ.get("FFMPEG_BIN") or (str(_GYAN) if _GYAN.exists() else "ffmpeg")


def inject_demo_banner(page) -> None:
    """Inject a crisp Ocean/Crusoe editorial chapter caption bar at the bottom-center for the video."""
    page.evaluate(
        """() => {
        if (document.getElementById('demoCaptionOverlay')) return;
        const bar = document.createElement('div');
        bar.id = 'demoCaptionOverlay';
        bar.style.cssText = [
          'position:fixed',
          'bottom:62px',
          'left:50%',
          'transform:translateX(-50%)',
          'z-index:9999',
          'background:#0B1B2B',
          'color:#F8FBFF',
          'border:2px solid #3B9EFF',
          'box-shadow:5px 5px 0 #3B9EFF',
          'padding:8px 18px',
          'font-family:Manrope, system-ui, sans-serif',
          'font-size:14px',
          'font-weight:800',
          'letter-spacing:0.04em',
          'pointer-events:none',
          'white-space:nowrap',
          'transition:opacity 0.2s ease'
        ].join(';');
        bar.textContent = 'KRONOS-1M // 1M-Context Robotics Data Engine & Multiverse DPO Arena';
        document.body.appendChild(bar);
    }"""
    )


def set_caption(page, stage: str, text: str) -> None:
    page.evaluate(
        """([st, txt]) => {
        const el = document.getElementById('demoCaptionOverlay');
        if (!el) return;
        el.innerHTML = `<span style="background:#3B9EFF;color:#0B1B2B;padding:2px 8px;margin-right:10px;font-weight:900;text-transform:uppercase;font-size:11px;letter-spacing:0.12em">${st}</span>${txt}`;
    }""",
        [stage, text],
    )


def main() -> int:
    url = os.environ.get("KRONOS_URL", "http://127.0.0.1:5173/")
    print(f"[Kronos Demo Recorder] Launching headless Chromium at 1920x1080 -> {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(OUT),
            record_video_size={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2500)
        inject_demo_banner(page)

        t_start = time.time()
        print("[00.0s] Stage 1: Run & Label View (Live 3D Rover + 6-Stage Pipeline + 0.5s Strip)")
        set_caption(
            page,
            "01 · LIVE 1M DATA ENGINE",
            "Continuous 3D Simulation, 32-Beam LiDAR, 0.5s Dense Process Rewards & VLA Action Captions",
        )
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / "01-run-and-label.png"))

        # Switch camera to Orbit and back to Chase to show 3D viewport controls
        set_caption(
            page,
            "01 · 3D SENSOR SUITE",
            "Tactical Orbit & FLIR / NVG Multimodal Vision + 60Hz Rapier3D Physics Telemetry",
        )
        page.locator("button:has-text('ORBIT')").click()
        page.wait_for_timeout(2500)
        page.locator("button:has-text('CHASE')").click()
        page.wait_for_timeout(2000)

        # [10s - 20s] Inject Chasm Curveball, toggle 64K Fractured Baseline, & Fork 4x Multiverse
        print(f"[{time.time() - t_start:04.1f}s] Stage 2: Curveball Hazard & 64K Fractured Seam vs 1M Context")
        set_caption(
            page,
            "02 · CURVEBALL & 64K SEAM TEAR",
            "Injecting Chasm Rupture (ΔY -2.4m) & Comparing 64K Chunked Seam Fracture vs 1M Unified Context",
        )
        page.locator("#btnInjectCurveball").click()
        page.wait_for_timeout(2500)
        page.locator("#btnToggleBaseline").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "02-64k-fractured-seam.png"))
        page.locator("#btnToggleBaseline").click()
        page.wait_for_timeout(1500)

        # Trigger 4-Branch Counterfactual Multiverse Fork in 3D
        set_caption(
            page,
            "03 · 4-WAY MULTIVERSE DPO FORK",
            "Escalating thinking_level: HIGH -> Spawning 4 Counterfactual Ghost Rollouts (α, β, γ, Ω) in 3D",
        )
        page.locator("#btnForkMultiverse").click()
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / "03-multiverse-ghosts-3d.png"))

        # [20s - 30s] Tab 2: Data & Captions (Temporal Segmentation + 0.5s Process Rewards + Rubric)
        print(f"[{time.time() - t_start:04.1f}s] Stage 3: Tab 2 - Data & Captions + Test-Time Rubric")
        page.locator("#tab-labels").click()
        set_caption(
            page,
            "04 · TEMPORAL SEGMENTS & RUBRIC",
            "WGO-Bench Sub-Task Segmentation (54.8% F1), VLA Captions (81.2% Acc) & Zero-Shot Rubric Grading",
        )
        page.wait_for_timeout(3500)
        page.locator("button:has-text('Strict Slip (< 15%)')").click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "04-data-labels-and-rubric.png"))
        page.wait_for_timeout(2000)

        # [30s - 40s] Tab 3: Multiverse DPO & Hardware Co-Design
        print(f"[{time.time() - t_start:04.1f}s] Stage 4: Tab 3 - Multiverse DPO Corrections & Co-Design")
        page.locator("#tab-multiverse").click()
        set_caption(
            page,
            "05 · COUNTERFACTUAL DPO & CO-DESIGN",
            "Chosen Ghost Ω (4.9/5) vs 3 Rejected Rollouts + Live Thruster Hopper Hardware Co-Design",
        )
        page.wait_for_timeout(3500)
        page.locator("button:has-text('+ Morph: Thruster Hopper')").click()
        page.wait_for_timeout(2500)
        page.locator("#btnDeployBridge").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "05-multiverse-dpo-codesign.png"))

        # [40s - 50s] Tab 4: Video Checks & Tab 5: GPU Train & Sim Verification
        print(f"[{time.time() - t_start:04.1f}s] Stage 5: Tab 4 - Video Checks & Tab 5 - GPU DPO Training")
        page.locator("#tab-check").click()
        set_caption(
            page,
            "06 · RAW .MP4 VIDEO CHECKS",
            "4.81 MB Raw .MP4 Video Evaluated by Gemini 4 Argon: 100% Within-1 Accuracy (MAE×100 = 12.5)",
        )
        page.wait_for_timeout(4500)
        page.screenshot(path=str(OUT / "06-video-checks.png"))

        page.locator("#tab-train").click()
        set_caption(
            page,
            "07 · BREV & RTX 3080 CUDA DPO TRAIN",
            "Contrastive DPO Training (Loss 0.693 -> 0.142) · Post-DPO Sim Survival Jumps 0.0% -> 98.2%",
        )
        page.locator("button:has-text('Run Contrastive DPO Training')").click()
        page.wait_for_timeout(5500)
        page.screenshot(path=str(OUT / "07-gpu-dpo-training.png"))

        # [50s - 60s] Tab 6: FLORA vs GPT-6 Astra Benchmark + Crusoe Theme + Final Run View
        print(f"[{time.time() - t_start:04.1f}s] Stage 6: Tab 6 - Official FLORA vs Astra Benchmark & Crusoe Theme")
        page.locator("#tab-benchmark").click()
        set_caption(
            page,
            "08 · KRONOS-1M vs FLORA & GPT-6 ASTRA",
            "94.4% Failure Recall (vs Astra 0.0%), 0.0% False Alarm (vs FLORA 11.0%), $1.18/hr (95% Cache)",
        )
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "08-flora-astra-benchmark.png"))

        # Show Crusoe Olive/Lime Theme switch on Run view for final seconds
        page.locator("#tab-run").click()
        page.locator("#btnThemeToggle").click()
        set_caption(
            page,
            "09 · CRUSOE + OCEAN DUAL THEME",
            "KRONOS-1M Autonomous Physical Auto-Research & Counterfactual Data Engine — Ready for Deployment",
        )
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / "09-crusoe-theme-run.png"))
        page.locator("#btnThemeToggle").click()

        # Pad to ~60s total recording duration
        elapsed = time.time() - t_start
        remaining_ms = max(1000, int((60.0 - elapsed) * 1000))
        print(f"[{elapsed:04.1f}s] Holding final live simulation frame for {remaining_ms}ms to hit ~60s...")
        page.wait_for_timeout(remaining_ms)

        video = page.video
        context.close()
        browser.close()

        if video is not None:
            webm_src = Path(video.path())
            webm_dest = OUT / "kronos-1m-demo.webm"
            if webm_dest.exists():
                webm_dest.unlink()
            shutil.move(str(webm_src), str(webm_dest))
            print(f"[Kronos Demo Recorder] Saved WebM: {webm_dest} ({webm_dest.stat().st_size / 1e6:.2f} MB)")

            mp4_dest = OUT / "kronos-1m-demo.mp4"
            cmd = [
                FFMPEG,
                "-y",
                "-i",
                str(webm_dest),
                "-t",
                "60",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(mp4_dest),
            ]
            print(f"[Kronos Demo Recorder] Transcoding to H.264 MP4 via {FFMPEG}...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and mp4_dest.exists():
                print(f"[Kronos Demo Recorder] Saved 60s MP4: {mp4_dest} ({mp4_dest.stat().st_size / 1e6:.2f} MB)")
            else:
                print(f"[Kronos Demo Recorder] ffmpeg warning: {r.stderr[-400:]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
