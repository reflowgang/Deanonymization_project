#!/usr/bin/env python3
"""P1 Reason concurrency bench — reuse baseline packs, vary --concurrency.

Reuses extract summaries + search_top15 (+ candidate summaries) from an existing
regression_50 run. Does NOT re-extract. Measures wall time / per-user latency
and error rates for sequential vs concurrent Reason calls to local vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from esrc.config import load_dotenv_files  # noqa: E402
from esrc.generate import generate, get_client  # noqa: E402
from esrc.manifests import git_commit, write_manifest  # noqa: E402
from esrc.paths import (  # noqa: E402
    bsp_pool_en,
    bsp_prompt_record_selection,
    fixture_user_ids,
    summer_root,
)
from esrc.reason_prompt import (  # noqa: E402
    build_candidate_block,
    build_user_prompt,
    extract_json_object,
    load_prompt_template as load_reason_template,
    resolve_predicted_candidate_id,
)

DEFAULT_REASON_MODEL = "qwen3.6-35b-a3b-nvfp4"
DEFAULT_SOURCE = (
    summer_root() / "results" / "runs" / "p1_baseline_regression_50_20260805"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reason concurrency bench (no re-extract)")
    p.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Existing baseline run with extract/ + search_top15.csv",
    )
    p.add_argument("--fixture-id", default="regression_50")
    p.add_argument("--level", default="T8")
    p.add_argument("--reason-model", default=os.getenv("VLLM_REASON_MODEL", DEFAULT_REASON_MODEL))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=10, help="Number of fixture users (default 10)")
    p.add_argument(
        "--concurrency",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    p.add_argument("--reason-max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout seconds per request")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--stop-on-error-rate",
        type=float,
        default=0.5,
        help="Skip higher K if error rate at current K exceeds this (0–1)",
    )
    return p.parse_args()


def parse_concurrencies(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise SystemExit(f"concurrency must be >= 1, got {k}")
        vals.append(k)
    if not vals:
        raise SystemExit("No concurrency levels provided")
    return vals


def load_search_by_query(search_path: Path) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = {}
    with search_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_query.setdefault(row["query_user_id"], []).append(row)
    return by_query


def build_pack(
    uid: str,
    *,
    extract_dir: Path,
    by_query: dict[str, list[dict]],
    cand_summ_dir: Path,
    reason_tpl: str,
) -> dict[str, Any]:
    rows = sorted(by_query.get(uid, []), key=lambda r: int(r["rank"]))
    qsum_path = extract_dir / f"{uid}.txt"
    if not qsum_path.exists():
        raise FileNotFoundError(f"Missing extract summary: {qsum_path}")
    q_text = qsum_path.read_text(encoding="utf-8").strip()
    cands = []
    for r in rows[:15]:
        cid = r["candidate_user_id"]
        cpath = cand_summ_dir / f"{cid}.txt"
        cands.append(
            {
                "candidate_user_id": cid,
                "rank": int(r["rank"]),
                "score": float(r["score"]),
                "summary": cpath.read_text(encoding="utf-8").strip(),
            }
        )
    block = build_candidate_block(cands)
    user_prompt = build_user_prompt(reason_tpl, q_text, block)
    return {
        "query_user_id": uid,
        "user_prompt": user_prompt,
        "candidate_ids": [c["candidate_user_id"] for c in cands],
        "n_candidates": len(cands),
    }


def reason_one(
    pack: dict[str, Any],
    *,
    reason_model: str,
    seed: int,
    max_tokens: int,
    timeout: float,
    fixture_id: str,
    level: str,
) -> dict[str, Any]:
    uid = pack["query_user_id"]
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "fixture_id": fixture_id,
        "T_level": level,
        "query_user_id": uid,
        "status": "error",
        "error": "",
        "error_type": "",
        "model": reason_model,
        "latency_s": None,
    }
    try:
        # One client per call keeps ThreadPoolExecutor isolation simple.
        client = get_client(timeout=timeout)
        result = generate(
            [{"role": "user", "content": pack["user_prompt"]}],
            model=reason_model,
            client=client,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=False,
        )
        obj = extract_json_object(result.text)
        pred_id, pred_num, err = resolve_predicted_candidate_id(obj, pack["candidate_ids"])
        if err:
            raise ValueError(err)
        row.update(
            {
                "status": "ok",
                "selected_candidate_user_id": pred_id,
                "selected_candidate_number": pred_num,
                "correct": pred_id == uid,
                "true_in_top15": uid in pack["candidate_ids"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["error_type"] = type(exc).__name__
        row["error"] = f"{type(exc).__name__}: {exc}"
        # Keep short traceback for 500 / connection diagnosis without huge logs.
        row["traceback_tail"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    row["latency_s"] = round(time.perf_counter() - t0, 3)
    return row


def run_level(
    packs: list[dict[str, Any]],
    *,
    concurrency: int,
    reason_model: str,
    seed: int,
    max_tokens: int,
    timeout: float,
    fixture_id: str,
    level: str,
    out_subdir: Path,
) -> dict[str, Any]:
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_subdir / "reason_predictions.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    wall_t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []

    if concurrency == 1:
        for pack in packs:
            row = reason_one(
                pack,
                reason_model=reason_model,
                seed=seed,
                max_tokens=max_tokens,
                timeout=timeout,
                fixture_id=fixture_id,
                level=level,
            )
            rows.append(row)
            status = row["status"]
            lat = row["latency_s"]
            print(f"  K=1 {row['query_user_id']} {status} {lat:.1f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    reason_one,
                    pack,
                    reason_model=reason_model,
                    seed=seed,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    fixture_id=fixture_id,
                    level=level,
                ): pack["query_user_id"]
                for pack in packs
            }
            for fut in as_completed(futures):
                uid = futures[fut]
                row = fut.result()
                rows.append(row)
                print(
                    f"  K={concurrency} {uid} {row['status']} {row['latency_s']:.1f}s",
                    flush=True,
                )

    wall_s = time.perf_counter() - wall_t0
    # Stable order for jsonl
    order = {p["query_user_id"]: i for i, p in enumerate(packs)}
    rows.sort(key=lambda r: order.get(r["query_user_id"], 10**9))

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_err = n - n_ok
    error_types: dict[str, int] = {}
    for r in rows:
        if r["status"] != "ok":
            et = r.get("error_type") or "Unknown"
            error_types[et] = error_types.get(et, 0) + 1

    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    summary = {
        "concurrency": concurrency,
        "n_users": n,
        "n_ok": n_ok,
        "n_error": n_err,
        "error_rate": (n_err / n) if n else 0.0,
        "error_types": error_types,
        "wall_s": round(wall_s, 3),
        "wall_min": round(wall_s / 60.0, 3),
        "effective_min_per_user": round((wall_s / 60.0) / n, 4) if n else None,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else None,
        "max_latency_s": round(max(latencies), 3) if latencies else None,
        "min_latency_s": round(min(latencies), 3) if latencies else None,
        "speedup_vs_sum_latency": (
            round(sum(latencies) / wall_s, 3) if wall_s > 0 and latencies else None
        ),
        "projected_hours_500": round((wall_s / n) * 500 / 3600, 2) if n else None,
        "projected_hours_1000": round((wall_s / n) * 1000 / 3600, 2) if n else None,
        "user_ids": [r["query_user_id"] for r in rows],
        "per_user": [
            {
                "query_user_id": r["query_user_id"],
                "status": r["status"],
                "latency_s": r["latency_s"],
                "error_type": r.get("error_type") or None,
                "correct": r.get("correct"),
            }
            for r in rows
        ],
    }
    (out_subdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    load_dotenv_files(summer_root())
    # Allow explicit override from shell even if .env missing (common for remote GPU).
    if not os.getenv("VLLM_BASE_URL"):
        print("WARN: VLLM_BASE_URL unset; set it before running against gpu6", flush=True)

    args = parse_args()
    concurrencies = parse_concurrencies(args.concurrency)
    users = fixture_user_ids(args.fixture_id)[: args.limit]
    if len(users) < args.limit:
        print(f"WARN: only {len(users)} fixture users (requested {args.limit})")

    source = args.source_dir
    extract_dir = source / "extract"
    search_path = source / "search_top15.csv"
    if not extract_dir.is_dir():
        raise SystemExit(f"Missing extract dir: {extract_dir}")
    if not search_path.exists():
        raise SystemExit(f"Missing search csv: {search_path}")

    out_dir = args.out_dir or (
        summer_root()
        / "results"
        / "runs"
        / f"p1_reason_concurrency_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reason_tpl = load_reason_template(bsp_prompt_record_selection())
    by_query = load_search_by_query(search_path)
    cand_summ_dir = bsp_pool_en() / "candidate_summaries"

    packs = [
        build_pack(
            uid,
            extract_dir=extract_dir,
            by_query=by_query,
            cand_summ_dir=cand_summ_dir,
            reason_tpl=reason_tpl,
        )
        for uid in users
    ]
    print(
        f"Built {len(packs)} packs from {source}\n"
        f"Model={args.reason_model} enable_thinking=False\n"
        f"Concurrency levels: {concurrencies}\n"
        f"Out → {out_dir}",
        flush=True,
    )

    level_summaries: list[dict[str, Any]] = []
    skipped: list[int] = []
    for k in concurrencies:
        print(f"\n=== concurrency={k} ===", flush=True)
        summary = run_level(
            packs,
            concurrency=k,
            reason_model=args.reason_model,
            seed=args.seed,
            max_tokens=args.reason_max_tokens,
            timeout=args.timeout,
            fixture_id=args.fixture_id,
            level=args.level,
            out_subdir=out_dir / f"k{k}",
        )
        level_summaries.append(summary)
        print(
            f"  wall={summary['wall_s']:.1f}s "
            f"({summary['effective_min_per_user']:.3f} min/user) "
            f"ok={summary['n_ok']}/{summary['n_users']} "
            f"errors={summary['error_types']}",
            flush=True,
        )
        if summary["error_rate"] > args.stop_on_error_rate:
            remaining = [x for x in concurrencies if x > k]
            if remaining:
                print(
                    f"  STOP: error_rate={summary['error_rate']:.0%} > "
                    f"{args.stop_on_error_rate:.0%}; skipping higher K={remaining}",
                    flush=True,
                )
                skipped.extend(remaining)
            break

    # Pick best stable K: lowest error_rate, then highest concurrency, then lowest wall.
    stable = [s for s in level_summaries if s["error_rate"] <= 0.1]
    if not stable:
        stable = [s for s in level_summaries if s["n_ok"] > 0] or level_summaries
    best = min(
        stable,
        key=lambda s: (s["error_rate"], -s["concurrency"], s["wall_s"]),
    )

    report = {
        "task": "p1_reason_concurrency_bench",
        "source_dir": str(source),
        "reason_model": args.reason_model,
        "enable_thinking": False,
        "n_users": len(packs),
        "user_ids": users,
        "concurrencies_requested": concurrencies,
        "concurrencies_skipped": skipped,
        "levels": level_summaries,
        "best_stable": {
            "concurrency": best["concurrency"],
            "wall_s": best["wall_s"],
            "effective_min_per_user": best["effective_min_per_user"],
            "error_rate": best["error_rate"],
            "projected_hours_500": best["projected_hours_500"],
            "projected_hours_1000": best["projected_hours_1000"],
        },
        "git_commit": git_commit(summer_root()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Human-readable table
    lines = [
        "K\twall_s\tmin/user\terrors\terr_rate\tproj_h_500\tproj_h_1000",
    ]
    for s in level_summaries:
        lines.append(
            f"{s['concurrency']}\t{s['wall_s']:.1f}\t{s['effective_min_per_user']:.3f}\t"
            f"{s['n_error']}/{s['n_users']}\t{s['error_rate']:.0%}\t"
            f"{s['projected_hours_500']}\t{s['projected_hours_1000']}"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "comparison.tsv").write_text(table, encoding="utf-8")
    print("\n" + table, flush=True)
    print(
        f"Best stable K={best['concurrency']}: "
        f"{best['effective_min_per_user']:.3f} min/user → "
        f"~{best['projected_hours_500']}h / 500, "
        f"~{best['projected_hours_1000']}h / 1000",
        flush=True,
    )

    write_manifest(
        out_dir / "manifest.json",
        {
            "task": "p1_reason_concurrency_bench",
            "source_dir": str(source),
            "reason_model": args.reason_model,
            "n_users": len(packs),
            "concurrencies": concurrencies,
            "report": report,
            "git_commit": git_commit(summer_root()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
