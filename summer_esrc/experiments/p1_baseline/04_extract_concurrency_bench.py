#!/usr/bin/env python3
"""P1 Extract concurrency bench — proportional chunked/single mix, vary K.

Selects ~10 users from a prior extract_meta.jsonl with intentional mix
(~66% chunked ≈ 7 chunked + 3 single-pass). Re-runs Extract on truncated
queries at concurrency K∈{1,2,4,8}. Does NOT run Search/Reason or full pool.
"""

from __future__ import annotations

import argparse
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
from esrc.extract import extract_profile, load_prompt_template  # noqa: E402
from esrc.generate import get_client  # noqa: E402
from esrc.manifests import git_commit, write_manifest  # noqa: E402
from esrc.paths import (  # noqa: E402
    bsp_prompt_summarization,
    bsp_truncated_query,
    summer_prompt_extract_merge,
    summer_root,
)

DEFAULT_EXTRACT_MODEL = "qwen3.5-4b"
DEFAULT_SOURCE = (
    summer_root() / "results" / "runs" / "p1_baseline_regression_50_20260805"
)
# Intentional 7 chunked + 3 single ≈ 33:17 from regression_50 meta.
# Chunked mix spans 2/3/4 chunks (representative; omits 16-chunk outlier for
# projection fairness — still stresses multi-call concurrent users).
DEFAULT_CHUNKED_IDS = [
    "user_043ea160",  # 4 chunks
    "user_476670f9",  # 4 chunks
    "user_30802254",  # 3 chunks
    "user_126670d7",  # 2 chunks
    "user_13fb2e34",  # 2 chunks
    "user_16b87e34",  # 2 chunks
    "user_1b2d69c1",  # 2 chunks
]
DEFAULT_SINGLE_IDS = [
    "user_159646d5",  # single-pass
    "user_19641691",  # single-pass
    "user_1d775dd8",  # single-pass
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract concurrency bench (no full pool)")
    p.add_argument(
        "--source-meta",
        type=Path,
        default=DEFAULT_SOURCE / "extract_meta.jsonl",
        help="Prior extract_meta.jsonl used to validate chunked/single labels",
    )
    p.add_argument("--level", default="T8")
    p.add_argument(
        "--extract-model",
        default=os.getenv("VLLM_EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL),
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--n-chunked",
        type=int,
        default=7,
        help="Target chunked users (default 7 ≈ 66%% of 10)",
    )
    p.add_argument(
        "--n-single",
        type=int,
        default=3,
        help="Target single-pass users (default 3)",
    )
    p.add_argument(
        "--concurrency",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    p.add_argument("--extract-max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout per request")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--stop-on-error-rate",
        type=float,
        default=0.5,
        help="Skip higher K if error rate at current K exceeds this (0–1)",
    )
    p.add_argument(
        "--user-ids",
        type=str,
        default="",
        help="Optional comma-separated override (skips proportional selection)",
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


def load_meta(path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("status") == "ok" and obj.get("user_id"):
                meta[obj["user_id"]] = obj
    return meta


def select_users(
    meta: dict[str, dict[str, Any]],
    *,
    n_chunked: int,
    n_single: int,
    override: list[str] | None,
) -> list[dict[str, Any]]:
    """Return selection rows with expected_chunked from prior meta."""
    if override:
        rows = []
        for uid in override:
            m = meta.get(uid)
            if m is None:
                raise SystemExit(f"user {uid} not in source meta {len(meta)} ok rows")
            rows.append(
                {
                    "user_id": uid,
                    "expected_chunked": bool(m.get("chunked")),
                    "expected_n_chunks": m.get("n_chunks"),
                    "prior_n_words": m.get("n_words"),
                    "prior_probe_tokens_full": m.get("probe_tokens_full"),
                }
            )
        return rows

    chunked = [uid for uid, m in meta.items() if m.get("chunked")]
    single = [uid for uid, m in meta.items() if not m.get("chunked")]

    # Prefer curated defaults when present in meta; fill from remaining.
    pick_c = [u for u in DEFAULT_CHUNKED_IDS if u in meta]
    pick_s = [u for u in DEFAULT_SINGLE_IDS if u in meta]
    for u in sorted(chunked):
        if u not in pick_c:
            pick_c.append(u)
    for u in sorted(single):
        if u not in pick_s:
            pick_s.append(u)

    if len(pick_c) < n_chunked:
        raise SystemExit(f"Need {n_chunked} chunked users, only {len(pick_c)} in meta")
    if len(pick_s) < n_single:
        raise SystemExit(f"Need {n_single} single users, only {len(pick_s)} in meta")

    selected = pick_c[:n_chunked] + pick_s[:n_single]
    rows = []
    for uid in selected:
        m = meta[uid]
        rows.append(
            {
                "user_id": uid,
                "expected_chunked": bool(m.get("chunked")),
                "expected_n_chunks": m.get("n_chunks"),
                "prior_n_words": m.get("n_words"),
                "prior_probe_tokens_full": m.get("probe_tokens_full"),
            }
        )
    return rows


def extract_one(
    sel: dict[str, Any],
    *,
    level: str,
    extract_model: str,
    extract_tpl: str,
    merge_tpl: str,
    max_tokens: int,
    timeout: float,
    seed: int | None,
) -> dict[str, Any]:
    uid = sel["user_id"]
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "user_id": uid,
        "T_level": level,
        "expected_chunked": sel["expected_chunked"],
        "expected_n_chunks": sel.get("expected_n_chunks"),
        "status": "error",
        "error": "",
        "error_type": "",
        "chunked": None,
        "n_chunks": None,
        "method": None,
        "latency_s": None,
    }
    try:
        qpath = bsp_truncated_query(level, uid)
        if not qpath.exists():
            raise FileNotFoundError(f"Missing truncated query: {qpath}")
        text = qpath.read_text(encoding="utf-8")
        row["n_words"] = len(text.split())
        # Per-call client: safe under ThreadPoolExecutor.
        client = get_client(timeout=timeout)
        result = extract_profile(
            text,
            client=client,
            model=extract_model,
            extract_template=extract_tpl,
            merge_template=merge_tpl,
            max_tokens=max_tokens,
            enable_thinking=False,
            seed=seed,
        )
        row.update(
            {
                "status": "ok",
                "chunked": result.chunked,
                "n_chunks": result.n_chunks,
                "method": result.method,
                "chunk_sizes": list(result.chunk_sizes),
                "probe_tokens_full": result.probe_tokens_full,
                "probe_tokens_per_chunk": list(result.probe_tokens_per_chunk),
                "summary_chars": len(result.summary),
                "chunk_flag_match": result.chunked == bool(sel["expected_chunked"]),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["error_type"] = type(exc).__name__
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback_tail"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
    row["latency_s"] = round(time.perf_counter() - t0, 3)
    return row


def _split_stats(rows: list[dict[str, Any]], *, chunked: bool) -> dict[str, Any]:
    subset = [r for r in rows if r.get("status") == "ok" and bool(r.get("chunked")) == chunked]
    # Fall back to expected flag if status ok but chunked missing (shouldn't happen)
    if not subset:
        subset = [
            r
            for r in rows
            if r.get("status") == "ok" and bool(r.get("expected_chunked")) == chunked
        ]
    lats = [float(r["latency_s"]) for r in subset if r.get("latency_s") is not None]
    return {
        "n": len(subset),
        "mean_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
        "median_latency_s": round(sorted(lats)[len(lats) // 2], 3) if lats else None,
        "max_latency_s": round(max(lats), 3) if lats else None,
        "min_latency_s": round(min(lats), 3) if lats else None,
        "user_ids": [r["user_id"] for r in subset],
    }


def run_level(
    selection: list[dict[str, Any]],
    *,
    concurrency: int,
    level: str,
    extract_model: str,
    extract_tpl: str,
    merge_tpl: str,
    max_tokens: int,
    timeout: float,
    seed: int | None,
    out_subdir: Path,
) -> dict[str, Any]:
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_subdir / "extract_meta.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    wall_t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []

    def _call(sel: dict[str, Any]) -> dict[str, Any]:
        return extract_one(
            sel,
            level=level,
            extract_model=extract_model,
            extract_tpl=extract_tpl,
            merge_tpl=merge_tpl,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
        )

    if concurrency == 1:
        for sel in selection:
            row = _call(sel)
            rows.append(row)
            print(
                f"  K=1 {row['user_id']} {row['status']} "
                f"chunked={row.get('chunked')} n_chunks={row.get('n_chunks')} "
                f"{row['latency_s']:.1f}s",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_call, sel): sel["user_id"] for sel in selection}
            for fut in as_completed(futures):
                uid = futures[fut]
                row = fut.result()
                rows.append(row)
                print(
                    f"  K={concurrency} {uid} {row['status']} "
                    f"chunked={row.get('chunked')} n_chunks={row.get('n_chunks')} "
                    f"{row['latency_s']:.1f}s",
                    flush=True,
                )

    wall_s = time.perf_counter() - wall_t0
    order = {s["user_id"]: i for i, s in enumerate(selection)}
    rows.sort(key=lambda r: order.get(r["user_id"], 10**9))

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
    chunked_stats = _split_stats(rows, chunked=True)
    single_stats = _split_stats(rows, chunked=False)

    # Effective rate from wall clock (accounts for overlap under concurrency).
    wall_per_user = (wall_s / n) if n else None
    # Mix-adjusted projection: weight by ~66% chunked / 34% single using
    # observed mean latencies, then scale by concurrency efficiency.
    mix_proj: dict[str, Any] = {"assumption": "66% chunked / 34% single from fixture"}
    if (
        chunked_stats["mean_latency_s"] is not None
        and single_stats["mean_latency_s"] is not None
        and wall_s > 0
        and latencies
    ):
        mix_mean_lat = 0.66 * chunked_stats["mean_latency_s"] + 0.34 * single_stats[
            "mean_latency_s"
        ]
        # Concurrency efficiency vs sum of per-user latencies.
        eff = sum(latencies) / wall_s if wall_s > 0 else 1.0
        # Project wall ≈ (mix_mean_lat * N) / eff
        mix_wall_per_user = mix_mean_lat / eff if eff > 0 else mix_mean_lat
        mix_proj.update(
            {
                "mix_mean_latency_s": round(mix_mean_lat, 3),
                "concurrency_efficiency": round(eff, 3),
                "effective_wall_s_per_user": round(mix_wall_per_user, 3),
                "projected_hours_500": round(mix_wall_per_user * 500 / 3600, 2),
                "projected_hours_1000": round(mix_wall_per_user * 1000 / 3600, 2),
            }
        )

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
        "effective_s_per_user": round(wall_per_user, 3) if wall_per_user else None,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else None,
        "max_latency_s": round(max(latencies), 3) if latencies else None,
        "min_latency_s": round(min(latencies), 3) if latencies else None,
        "speedup_vs_sum_latency": (
            round(sum(latencies) / wall_s, 3) if wall_s > 0 and latencies else None
        ),
        "projected_hours_500": round(wall_per_user * 500 / 3600, 2) if wall_per_user else None,
        "projected_hours_1000": round(wall_per_user * 1000 / 3600, 2) if wall_per_user else None,
        "chunked_ok": chunked_stats,
        "single_ok": single_stats,
        "mix66_projection": mix_proj,
        "user_ids": [r["user_id"] for r in rows],
        "per_user": [
            {
                "user_id": r["user_id"],
                "status": r["status"],
                "expected_chunked": r.get("expected_chunked"),
                "chunked": r.get("chunked"),
                "n_chunks": r.get("n_chunks"),
                "latency_s": r["latency_s"],
                "error_type": r.get("error_type") or None,
            }
            for r in rows
        ],
    }
    (out_subdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    load_dotenv_files(summer_root())
    if not os.getenv("VLLM_BASE_URL"):
        print("WARN: VLLM_BASE_URL unset; set it before running against gpu6", flush=True)

    args = parse_args()
    concurrencies = parse_concurrencies(args.concurrency)
    meta = load_meta(args.source_meta)
    if not meta:
        raise SystemExit(f"No ok rows in {args.source_meta}")

    override = [u.strip() for u in args.user_ids.split(",") if u.strip()] or None
    selection = select_users(
        meta,
        n_chunked=args.n_chunked,
        n_single=args.n_single,
        override=override,
    )
    n_exp_c = sum(1 for s in selection if s["expected_chunked"])
    n_exp_s = len(selection) - n_exp_c

    out_dir = args.out_dir or (
        summer_root()
        / "results"
        / "runs"
        / f"p1_extract_concurrency_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_tpl = load_prompt_template(bsp_prompt_summarization())
    merge_tpl = load_prompt_template(summer_prompt_extract_merge())

    sel_path = out_dir / "selection.json"
    sel_path.write_text(
        json.dumps(
            {
                "source_meta": str(args.source_meta),
                "n_chunked": n_exp_c,
                "n_single": n_exp_s,
                "mix_note": f"{n_exp_c}:{n_exp_s} ≈ fixture 33:17 chunked:single",
                "users": selection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Selected {len(selection)} users ({n_exp_c} chunked + {n_exp_s} single)\n"
        f"Model={args.extract_model} enable_thinking=False\n"
        f"Concurrency levels: {concurrencies}\n"
        f"Out → {out_dir}",
        flush=True,
    )
    for s in selection:
        print(
            f"  {s['user_id']} expected_chunked={s['expected_chunked']} "
            f"n_chunks={s.get('expected_n_chunks')} words={s.get('prior_n_words')}",
            flush=True,
        )

    level_summaries: list[dict[str, Any]] = []
    skipped: list[int] = []
    for k in concurrencies:
        print(f"\n=== concurrency={k} ===", flush=True)
        summary = run_level(
            selection,
            concurrency=k,
            level=args.level,
            extract_model=args.extract_model,
            extract_tpl=extract_tpl,
            merge_tpl=merge_tpl,
            max_tokens=args.extract_max_tokens,
            timeout=args.timeout,
            seed=args.seed,
            out_subdir=out_dir / f"k{k}",
        )
        level_summaries.append(summary)
        c = summary["chunked_ok"]
        s = summary["single_ok"]
        print(
            f"  wall={summary['wall_s']:.1f}s "
            f"({summary['effective_min_per_user']:.3f} min/user) "
            f"ok={summary['n_ok']}/{summary['n_users']} "
            f"errors={summary['error_types']}\n"
            f"  chunked mean_lat={c['mean_latency_s']}s (n={c['n']}) | "
            f"single mean_lat={s['mean_latency_s']}s (n={s['n']})",
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

    # Prefer lowest error, then lowest wall (throughput), then highest K as tiebreak.
    stable = [s for s in level_summaries if s["error_rate"] <= 0.1]
    if not stable:
        stable = [s for s in level_summaries if s["n_ok"] > 0] or level_summaries
    best = min(
        stable,
        key=lambda s: (s["error_rate"], s["wall_s"], -s["concurrency"]),
    )

    report = {
        "task": "p1_extract_concurrency_bench",
        "source_meta": str(args.source_meta),
        "extract_model": args.extract_model,
        "enable_thinking": False,
        "n_users": len(selection),
        "n_expected_chunked": n_exp_c,
        "n_expected_single": n_exp_s,
        "user_ids": [s["user_id"] for s in selection],
        "selection": selection,
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
            "mix66_projection": best.get("mix66_projection"),
            "chunked_ok": best.get("chunked_ok"),
            "single_ok": best.get("single_ok"),
        },
        "git_commit": git_commit(summer_root()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "K\twall_s\tmin/user\terrors\terr_rate\tproj_h_500\tproj_h_1000\t"
        "chunked_mean_s\tsingle_mean_s\tmix66_h_500\tmix66_h_1000",
    ]
    for s in level_summaries:
        mix = s.get("mix66_projection") or {}
        lines.append(
            f"{s['concurrency']}\t{s['wall_s']:.1f}\t{s['effective_min_per_user']:.3f}\t"
            f"{s['n_error']}/{s['n_users']}\t{s['error_rate']:.0%}\t"
            f"{s['projected_hours_500']}\t{s['projected_hours_1000']}\t"
            f"{s['chunked_ok']['mean_latency_s']}\t{s['single_ok']['mean_latency_s']}\t"
            f"{mix.get('projected_hours_500')}\t{mix.get('projected_hours_1000')}"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "comparison.tsv").write_text(table, encoding="utf-8")
    print("\n" + table, flush=True)
    print(
        f"Best stable K={best['concurrency']}: "
        f"{best['effective_min_per_user']:.3f} min/user → "
        f"~{best['projected_hours_500']}h / 500, "
        f"~{best['projected_hours_1000']}h / 1000 "
        f"(batch mix; mix66 → {best.get('mix66_projection', {}).get('projected_hours_1000')}h @1000)",
        flush=True,
    )

    write_manifest(
        out_dir / "manifest.json",
        {
            "task": "p1_extract_concurrency_bench",
            "source_meta": str(args.source_meta),
            "extract_model": args.extract_model,
            "n_users": len(selection),
            "n_chunked": n_exp_c,
            "n_single": n_exp_s,
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
