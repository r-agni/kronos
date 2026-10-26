#!/usr/bin/env python3
"""Score gemini-4-argon on teetone/RoboReward test without showing gold labels.

Reuses the existing generateContent client shape from gpuParallelExperiments.py
(model id, base URL, inline video/mp4, thinking_config). Gold reward, the
gpt5_mini_check rationale, and the video filename are never placed in the prompt.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "runs" / "benchmarks" / "_roboreward_test.parquet"
OUT = ROOT / "runs" / "benchmarks" / "honest_roboreward_eval.json"
CACHE = ROOT / "runs" / "benchmarks" / "_mp4_cache"
MODEL = "gemini-4-argon"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)
POPULATION_N = 2831
SAMPLE_N = 30
SEED = 0
# Pre-registered draw from Python 3.12 random.Random(0).sample(range(2831), 30).
# Not the previous hand-picked 8 (0, 49, 211, 400, 500, 1009, 1200, 1504).
EXPECTED_IDS = [
    1577, 1722, 165, 1060, 2094, 1990, 1658, 1242, 1952, 1466,
    2389, 894, 2067, 570, 1154, 572, 388, 2532, 1026, 2181,
    2465, 601, 1270, 404, 302, 2801, 1352, 1933, 2292, 412,
]
# Full split is both thinking levels. Above this estimated wall-clock
# (concurrency 4) we keep the pre-registered sample instead of 2,831.
FULL_SPLIT_HOUR_LIMIT = 2.0
# If one low call implies the remaining sample would take longer than this, stop.
LOW_REMAINING_MINUTE_LIMIT = 75.0
HIGH_TIMEOUT_S = 180
LOW_TIMEOUT_S = 150
MAX_VIDEO_BYTES = 18 * 1024 * 1024
WORKERS = 3

# Public rubric from Lee et al., arXiv:2601.00675, "Rubric and Validation".
RUBRIC = (
    "1 = No success: The final state shows no goal-relevant change for the task command.\n"
    "2 = Minimal progress: The final state shows a small but insufficient change toward the goal.\n"
    "3 = Partial completion: The final state shows good progress but violates major requirements or multiple requirements.\n"
    "4 = Near completion: The final state is correct in region and intent but misses a single minor requirement.\n"
    "5 = Perfect completion: The final state satisfies all the requirements."
)

_write_lock = threading.Lock()
_schema_lock = threading.Lock()
SCHEMA = "snake"


def load_env_key() -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY", "").strip()


def redact(text: str) -> str:
    text = re.sub(r"(key=)[^&\s\"']+", r"\1REDACTED", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "REDACTED", text)
    return text[:2000]


def build_prompt(task: str) -> str:
    prompt = (
        "You are scoring a real robot rollout video on the public RoboReward rubric.\n"
        f"Commanded task: {task}\n"
        "Watch the video from start to end. Judge only the final outcome against the task.\n"
        "Use exactly this rubric:\n"
        f"{RUBRIC}\n"
        'Respond with JSON only: {"predicted_score": 1|2|3|4|5, "visual_evidence": "one sentence describing the final state"}'
    )
    lowered = prompt.lower()
    for phrase in ("correct score", "provided score", "gold reward", "gpt5_mini_check"):
        if phrase in lowered:
            raise RuntimeError("prompt template contains a forbidden phrase")
    return prompt


def hf_to_https(path: str) -> str:
    match = re.match(r"hf://datasets/teetone/RoboReward@([^/]+)/(.+)$", path)
    if not match:
        raise ValueError(f"unexpected video path shape: {path[:80]}")
    rev, rel = match.group(1), match.group(2)
    return f"https://huggingface.co/datasets/teetone/RoboReward/resolve/{rev}/{rel}"


def load_rows() -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET, columns=["video", "task", "reward"])
    if table.num_rows != POPULATION_N:
        raise RuntimeError(f"expected {POPULATION_N} test rows, found {table.num_rows}")
    videos = table.column("video").to_pylist()
    tasks = table.column("task").to_pylist()
    rewards = table.column("reward").to_pylist()
    rows = []
    for i, (video, task, reward) in enumerate(zip(videos, tasks, rewards)):
        rows.append({
            "episode_id": i,
            "task": str(task),
            "gold": int(reward),
            "video_path": str(video["path"]),
        })
    return rows


def sample_ids() -> list[int]:
    ids = random.Random(SEED).sample(range(POPULATION_N), SAMPLE_N)
    if ids != EXPECTED_IDS:
        raise RuntimeError("seed-0 sample did not match the pre-registered id list")
    old = {0, 49, 211, 400, 500, 1009, 1200, 1504}
    if set(ids) == old or set(ids) <= old:
        raise RuntimeError("sample collapsed to the old hand-picked 8")
    return ids


def flush(doc: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, ensure_ascii=False)
    tmp = OUT.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    last_error: Exception | None = None
    for attempt in range(30):
        try:
            os.replace(tmp, OUT)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.15 * (attempt + 1))
    try:
        OUT.write_text(payload, encoding="utf-8")
    except PermissionError:
        if last_error:
            raise last_error
        raise
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def recompute(doc: dict) -> None:
    summary = {}
    for level in ("high", "low"):
        abs_errors = []
        exact = 0
        failed_ids = []
        attempted = 0
        prompt_tokens = 0
        candidate_tokens = 0
        thought_tokens = 0
        total_tokens = 0
        for ep in doc["episodes"]:
            call = (ep.get("calls") or {}).get(level)
            if call is None:
                continue
            attempted += 1
            usage = call.get("usage") or {}
            prompt_tokens += int(usage.get("promptTokenCount") or 0)
            candidate_tokens += int(usage.get("candidatesTokenCount") or 0)
            thought_tokens += int(usage.get("thoughtsTokenCount") or 0)
            total_tokens += int(usage.get("totalTokenCount") or 0)
            pred = call.get("pred")
            if pred is None:
                failed_ids.append(ep["episode_id"])
                continue
            err = abs(int(pred) - int(ep["gold"]))
            abs_errors.append(err)
            if err == 0:
                exact += 1
        n = len(abs_errors)
        summary[level] = {
            "n_attempted": attempted,
            "n_scored": n,
            "n_failed": len(failed_ids),
            "failed_episode_ids": failed_ids,
            "sum_abs_error": int(sum(abs_errors)),
            "mae": (sum(abs_errors) / n) if n else None,
            "mae_x100": (100.0 * sum(abs_errors) / n) if n else None,
            "exact_match": (exact / n) if n else None,
            "exact_match_count": exact,
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "thought_tokens": thought_tokens,
            "total_tokens": total_tokens,
            "status": "measured" if attempted else "not_measured",
            "benchmark_eligible": False,
        }
    doc["summary"] = summary


def init_doc(rows: list[dict], ids: list[int]) -> dict:
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing.get("protocol", {}).get("episode_ids") == ids:
            existing["status"] = "running"
            return existing
        raise RuntimeError(f"{OUT} exists with a different episode id list; refusing to overwrite")
    by_id = {r["episode_id"]: r for r in rows}
    episodes = []
    for eid in ids:
        row = by_id[eid]
        episodes.append({
            "episode_id": eid,
            "task": row["task"],
            "gold": row["gold"],
            "video_path": row["video_path"],
            "video_bytes": None,
            "video_sha256": None,
            "calls": {},
        })
    doc = {
        "status": "running",
        "protocol": {
            "dataset": "huggingface.co/datasets/teetone/RoboReward",
            "split": "test",
            "population_n": POPULATION_N,
            "result_label": "pre-registered random sample; not the paper's 2,831-episode MAE",
            "sample_n": SAMPLE_N,
            "seed": SEED,
            "rng": "Python 3.12 random.Random(0).sample(range(2831), 30)",
            "episode_ids": ids,
            "excluded_handpicked_8": [0, 49, 211, 400, 500, 1009, 1200, 1504],
            "overlap_with_handpicked_8": sorted(set(ids) & {0, 49, 211, 400, 500, 1009, 1200, 1504}),
            "model": MODEL,
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent",
            "thinking_levels_requested": ["high", "low"],
            "media": "dataset mp4 bytes as inline_data video/mp4",
            "prompt_contains": "commanded task + public 1-5 rubric from arXiv:2601.00675",
            "prompt_excludes": [
                "gold reward",
                "gpt5_mini_check rationale",
                "video filename and URL (791 test filenames contain a score token)",
            ],
            "rubric": RUBRIC,
            "mae_definition": "MAE x 100 = 100 * mean(abs(pred - gold)) over episodes with a parseable 1-5 score. Failures stay in n_failed and are not filled with gold.",
            "full_split_rule": (
                "After the first high and low calls, estimate full-split wall clock as "
                "(high_latency + low_latency) * 2831 / 3 / 3600. If that exceeds 2 hours, "
                "score only this sample."
            ),
            "dollar_cost": "not measured; no price field is returned by the API and no gemini-4-argon list price was applied",
        },
        "decision": {
            "full_split": None,
            "run_low": None,
            "reason": "pending first high call",
        },
        "episodes": episodes,
        "summary": {},
    }
    recompute(doc)
    return doc


def download_video(episode_id: int, video_path: str) -> tuple[bytes | None, str | None]:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{episode_id}.mp4"
    if dest.exists() and dest.stat().st_size > 32:
        data = dest.read_bytes()
    else:
        url = hf_to_https(video_path)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last_err = "download failed"
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = resp.read()
                break
            except Exception as exc:
                last_err = redact(str(exc))
                time.sleep(1 + attempt)
        if data is None:
            return None, last_err
        dest.write_bytes(data)
    if b"ftyp" not in data[:64]:
        return None, "download was not an mp4 (missing ftyp)"
    if len(data) > MAX_VIDEO_BYTES:
        return None, f"video_exceeds_inline_limit bytes={len(data)}"
    return data, None


def payload_for(video_b64: str, prompt: str, thinking_level: str, schema: str) -> dict:
    if schema == "snake":
        gen = {
            "responseMimeType": "application/json",
            "thinking_config": {"thinking_level": thinking_level},
        }
    else:
        gen = {
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": thinking_level},
        }
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": gen,
    }


def parse_score(text: str) -> int | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get("predicted_score", obj.get("score"))
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, str) and re.fullmatch(r"[1-5]", raw.strip()):
        return int(raw.strip())
    if isinstance(raw, float) and not raw.is_integer():
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value not in (1, 2, 3, 4, 5):
        return None
    return value


def extract_texts(body: dict) -> tuple[str, str, str | None]:
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback") or body.get("error")
        return "", "", redact(json.dumps(feedback))[:500] if feedback else "no candidates"
    parts = (candidates[0].get("content") or {}).get("parts") or []
    answer, thoughts = [], []
    for part in parts:
        text = part.get("text") or ""
        if part.get("thought") is True:
            thoughts.append(text)
        else:
            answer.append(text)
    finish = candidates[0].get("finishReason")
    err = None if answer else f"no answer text (finish={finish})"
    return "\n".join(answer).strip(), "\n".join(thoughts).strip(), err


def call_model(video: bytes, task: str, thinking_level: str) -> dict:
    global SCHEMA
    prompt = build_prompt(task)
    video_b64 = base64.b64encode(video).decode("ascii")
    api_key = load_env_key()
    if len(api_key) < 8:
        return {
            "thinking_level": thinking_level,
            "model": MODEL,
            "pred": None,
            "raw_model_text": "",
            "error": "missing GEMINI_API_KEY",
            "http_status": None,
            "usage": None,
            "latency_ms": 0,
            "schema": None,
        }
    timeout = HIGH_TIMEOUT_S if thinking_level == "high" else LOW_TIMEOUT_S
    last_status = None
    last_error = "unknown"
    raw_text = ""
    usage = None
    schema_used = None
    t0 = time.perf_counter()
    with _schema_lock:
        schemas = [SCHEMA, "camel" if SCHEMA == "snake" else "snake"]
    for schema in schemas:
        schema_used = schema
        body_bytes = json.dumps(payload_for(video_b64, prompt, thinking_level, schema)).encode("utf-8")
        req = urllib.request.Request(
            f"{ENDPOINT}?key={api_key}",
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    last_status = getattr(resp, "status", 200)
                    payload = json.loads(resp.read().decode("utf-8"))
                answer, thoughts, err = extract_texts(payload)
                raw_text = answer
                usage = payload.get("usageMetadata")
                pred = parse_score(answer) if answer else None
                if err and pred is None:
                    last_error = err
                else:
                    last_error = None if pred is not None else "no parseable predicted_score in model text"
                with _schema_lock:
                    SCHEMA = schema
                return {
                    "thinking_level": thinking_level,
                    "model": MODEL,
                    "pred": pred,
                    "raw_model_text": raw_text[:20000],
                    "raw_thought_excerpt": thoughts[:4000],
                    "error": last_error,
                    "http_status": last_status,
                    "usage": usage,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "schema": schema_used,
                    "finish_reason": (payload.get("candidates") or [{}])[0].get("finishReason"),
                }
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                detail = redact(exc.read().decode("utf-8", "replace"))
                last_error = f"http {exc.code}: {detail}"
                if exc.code in (401, 403, 404):
                    break
                if exc.code == 400 and schema == schemas[0]:
                    break
                if exc.code in (429, 500, 503):
                    time.sleep(2 + attempt * 3)
                    continue
                break
            except Exception as exc:
                last_error = redact(str(exc))
                time.sleep(1 + attempt)
        if last_status in (401, 403, 404):
            break
    return {
        "thinking_level": thinking_level,
        "model": MODEL,
        "pred": None,
        "raw_model_text": raw_text[:20000],
        "error": last_error,
        "http_status": last_status,
        "usage": usage,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "schema": schema_used,
    }


def episode_record(doc: dict, episode_id: int) -> dict:
    for ep in doc["episodes"]:
        if ep["episode_id"] == episode_id:
            return ep
    raise KeyError(episode_id)


def save_call(doc: dict, episode_id: int, level: str, call: dict, video: bytes | None, download_error: str | None) -> None:
    with _write_lock:
        ep = episode_record(doc, episode_id)
        if video is not None:
            ep["video_bytes"] = len(video)
            ep["video_sha256"] = hashlib.sha256(video).hexdigest()
        if download_error:
            ep["download_error"] = download_error
        ep["calls"][level] = call
        # Filename is stored for audit only. It is not sent to the model.
        ep["prompt_sent_gold"] = False
        ep["prompt_sent_filename"] = False
        recompute(doc)
        flush(doc)
    pred = call.get("pred")
    err = call.get("error")
    usage = call.get("usage") or {}
    print(
        f"EP_DONE id={episode_id} level={level} pred={pred} gold={episode_record(doc, episode_id)['gold']} "
        f"tokens={usage.get('totalTokenCount')} latency_ms={call.get('latency_ms')} error={err}",
        flush=True,
    )


def ensure_video(ep: dict) -> tuple[bytes | None, str | None]:
    if ep.get("download_error") and ep.get("video_bytes") is None:
        return None, ep["download_error"]
    return download_video(ep["episode_id"], ep["video_path"])


def score_level(doc: dict, ep: dict, level: str) -> dict:
    if level in ep.get("calls", {}):
        return ep["calls"][level]
    video, err = ensure_video(ep)
    if video is None:
        call = {
            "thinking_level": level,
            "model": MODEL,
            "pred": None,
            "raw_model_text": "",
            "error": err or "download failed",
            "http_status": None,
            "usage": None,
            "latency_ms": 0,
            "schema": None,
        }
        save_call(doc, ep["episode_id"], level, call, None, err)
        return call
    call = call_model(video, ep["task"], level)
    save_call(doc, ep["episode_id"], level, call, video, None)
    return call


def fatal_api(call: dict) -> bool:
    status = call.get("http_status")
    err = str(call.get("error") or "")
    if status in (401, 403, 404):
        return True
    if "missing GEMINI_API_KEY" in err:
        return True
    if "not found" in err.lower() and "model" in err.lower():
        return True
    return False


def decide_after_probe(doc: dict, high_call: dict, low_call: dict | None) -> None:
    latency_s = float(high_call.get("latency_ms") or 0) / 1000.0
    low_ok = bool(low_call) and low_call.get("http_status") == 200 and not fatal_api(low_call)
    low_s = float((low_call or {}).get("latency_ms") or 0) / 1000.0 if low_ok else latency_s
    per_episode_s = latency_s + low_s
    est_hours = per_episode_s * POPULATION_N / WORKERS / 3600.0
    run_full = est_hours <= FULL_SPLIT_HOUR_LIMIT and high_call.get("pred") is not None
    run_low = True
    low_reason = "low is a separate call on the same episodes"
    if low_call is None or fatal_api(low_call) or (low_call.get("pred") is None and (low_call.get("http_status") or 0) >= 400):
        run_low = False
        low_reason = "low call was rejected or unusable; low MAE will not be invented"
    else:
        remaining = (POPULATION_N if run_full else SAMPLE_N) - 1
        est_low_min = (low_s * remaining) / WORKERS / 60.0
        if est_low_min > LOW_REMAINING_MINUTE_LIMIT and not run_full:
            run_low = False
            low_reason = (
                f"remaining low calls estimated at {est_low_min:.1f} min "
                f"(limit {LOW_REMAINING_MINUTE_LIMIT}); low MAE not invented for the rest"
            )
        else:
            low_reason = f"remaining low calls estimated at {est_low_min:.1f} min"
    reason = (
        f"first high latency {latency_s:.1f}s; estimated full split "
        f"{est_hours:.2f} h at concurrency {WORKERS} (both levels). "
        + (
            "Running the full 2,831."
            if run_full
            else "Full split exceeds 2 hours, so the seed-0 sample of 30 is the reported set. "
        )
        + low_reason
    )
    doc["decision"] = {
        "full_split": run_full,
        "run_low": run_low,
        "estimated_full_split_hours": round(est_hours, 3),
        "probe_high_latency_s": round(latency_s, 3),
        "probe_low_latency_s": (
            round(float(low_call.get("latency_ms") or 0) / 1000.0, 3) if low_call else None
        ),
        "probe_high_usage": high_call.get("usage"),
        "probe_low_usage": (low_call or {}).get("usage"),
        "reason": reason,
    }
    print("PROBE_DECISION " + reason, flush=True)


def score_episode(doc: dict, episode_id: int, levels: list[str]) -> None:
    ep = episode_record(doc, episode_id)
    for level in levels:
        if level in ep.get("calls", {}):
            continue
        call = score_level(doc, ep, level)
        if fatal_api(call):
            doc["status"] = "stopped_api_error"
            with _write_lock:
                recompute(doc)
                flush(doc)
            print("FATAL_API " + redact(str(call.get("error"))), flush=True)
            return


def main() -> int:
    if not PARQUET.exists():
        print("missing parquet cache", flush=True)
        return 2
    ids = sample_ids()
    rows = load_rows()
    doc = init_doc(rows, ids)
    with _write_lock:
        flush(doc)
    print(f"PROTOCOL_LOCKED n={SAMPLE_N} seed={SEED} ids={ids}", flush=True)

    probe = episode_record(doc, ids[0])
    high_call = score_level(doc, probe, "high")
    if fatal_api(high_call):
        doc["status"] = "stopped_api_error"
        doc["decision"]["reason"] = "high call failed before a sample could be scored"
        doc["decision"]["run_low"] = False
        doc["decision"]["full_split"] = False
        with _write_lock:
            recompute(doc)
            flush(doc)
        print("FATAL_API " + redact(str(high_call.get("error"))), flush=True)
        return 1

    low_call = score_level(doc, probe, "low")
    decide_after_probe(doc, high_call, low_call)
    if fatal_api(low_call):
        doc["decision"]["run_low"] = False
    with _write_lock:
        recompute(doc)
        flush(doc)
    if doc.get("status") == "stopped_api_error":
        print("EVAL_COMPLETE status=stopped_api_error", flush=True)
        return 1

    if high_call.get("pred") is None and (high_call.get("http_status") or 0) >= 400:
        doc["status"] = "stopped_api_error"
        doc["decision"]["run_low"] = False
        doc["decision"]["full_split"] = False
        doc["decision"]["reason"] = (
            "probe high call returned HTTP "
            f"{high_call.get('http_status')} with no parseable score; "
            "remaining episodes were not scored and no MAE was invented"
        )
        with _write_lock:
            recompute(doc)
            flush(doc)
        print("FATAL_API " + redact(str(high_call.get("error"))), flush=True)
        return 1

    levels = ["high"] + (["low"] if doc["decision"].get("run_low") else [])
    reported_ids = ids
    if doc["decision"].get("full_split"):
        have = {ep["episode_id"] for ep in doc["episodes"]}
        for row in rows:
            if row["episode_id"] in have:
                continue
            doc["episodes"].append({
                "episode_id": row["episode_id"],
                "task": row["task"],
                "gold": row["gold"],
                "video_path": row["video_path"],
                "video_bytes": None,
                "video_sha256": None,
                "calls": {},
            })
        reported_ids = list(range(POPULATION_N))
        doc["protocol"]["result_label"] = "full teetone/RoboReward test split"
        doc["protocol"]["reported_n"] = POPULATION_N
        with _write_lock:
            flush(doc)
    else:
        doc["protocol"]["reported_n"] = SAMPLE_N

    remaining = []
    for eid in reported_ids:
        ep = episode_record(doc, eid)
        if any(level not in ep.get("calls", {}) for level in levels):
            remaining.append(eid)

    fatal = {"hit": False}

    def _run(eid: int) -> None:
        if fatal["hit"]:
            return
        before = len(episode_record(doc, eid).get("calls", {}))
        score_episode(doc, eid, levels)
        if doc.get("status") == "stopped_api_error":
            fatal["hit"] = True
        after = episode_record(doc, eid).get("calls", {})
        if len(after) == before and not fatal["hit"]:
            return

    if remaining:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(_run, eid) for eid in remaining]
            for fut in as_completed(futures):
                fut.result()
                if fatal["hit"]:
                    break

    doc["status"] = "complete" if doc.get("status") != "stopped_api_error" else doc["status"]
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _write_lock:
        recompute(doc)
        reported_set = set(reported_ids)
        for level, block in doc["summary"].items():
            covered = 0
            for ep in doc["episodes"]:
                if ep["episode_id"] in reported_set and level in (ep.get("calls") or {}):
                    covered += 1
            block["benchmark_eligible"] = covered == len(reported_set) and level in levels
            if not block["benchmark_eligible"]:
                block["status"] = "probe_only_not_a_benchmark" if covered else "not_measured"
        flush(doc)
    print("EVAL_COMPLETE status=" + doc["status"], flush=True)
    print(json.dumps(doc["summary"], indent=2), flush=True)
    return 0 if doc["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
