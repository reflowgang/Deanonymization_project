#!/usr/bin/env python3
"""P1 baseline — regression_50 end-to-end: Extract → Embed → Search → Reason.

Uses token-probe-gated Extract chunking. Compares metrics to fixture manifest
baselines (BSP gpt-4o reference) and flags chunking usage per user.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from esrc.config import load_dotenv_files  # noqa: E402
from esrc.embed import embed_texts  # noqa: E402
from esrc.extract import extract_profile, load_prompt_template  # noqa: E402
from esrc.generate import ContextLengthExceededError, PermanentRequestError, generate, get_client  # noqa: E402
from esrc.manifests import git_commit, sha256_file, write_manifest  # noqa: E402
from esrc.paths import (  # noqa: E402
    bsp_candidate_embeddings_index_csv,
    bsp_candidate_embeddings_npy,
    bsp_pool_en,
    bsp_prompt_record_selection,
    bsp_prompt_summarization,
    fixture_dir,
    fixture_user_ids,
    summer_prompt_extract_merge,
    summer_root,
)
from esrc.reason_prompt import (  # noqa: E402
    build_candidate_block,
    build_user_prompt,
    extract_json_object,
    load_prompt_template as load_reason_template,
    resolve_predicted_candidate_id,
)
from esrc.resume import load_last_ok_by_user, load_ok_user_ids, load_resume_skip_user_ids  # noqa: E402
from esrc.search import search_top_k  # noqa: E402

DEFAULT_EXTRACT_MODEL = "qwen3.5-4b"
DEFAULT_REASON_MODEL = "qwen3.6-35b-a3b-nvfp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="regression_50 E→S→R baseline with chunking")
    p.add_argument("--fixture-id", default="regression_50")
    p.add_argument("--level", default="T8")
    p.add_argument("--extract-model", default=os.getenv("VLLM_EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL))
    p.add_argument("--reason-model", default=os.getenv("VLLM_REASON_MODEL", DEFAULT_REASON_MODEL))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--phase", choices=("all", "extract", "search", "reason"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--extract-max-tokens", type=int, default=1024)
    p.add_argument("--reason-max-tokens", type=int, default=1024)
    return p.parse_args()


def load_candidate_index() -> tuple[list[str], np.ndarray]:
    idx_path = bsp_candidate_embeddings_index_csv()
    emb_path = bsp_candidate_embeddings_npy()
    user_ids: list[str] = []
    with idx_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user_ids.append(row["user_id"])
    matrix = np.load(emb_path)
    if matrix.shape[0] != len(user_ids):
        raise SystemExit(f"Embedding matrix rows {matrix.shape[0]} != index {len(user_ids)}")
    return user_ids, matrix


def phase_extract(
    users: list[str],
    *,
    level: str,
    client,
    extract_model: str,
    extract_tpl: str,
    merge_tpl: str,
    out_dir: Path,
    resume: bool,
    max_tokens: int,
) -> dict[str, dict]:
    extract_dir = out_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "extract_meta.jsonl"
    done = load_resume_skip_user_ids(meta_path) if resume else set()
    meta: dict[str, dict] = load_last_ok_by_user(meta_path) if resume else {}

    for uid in users:
        if uid in done:
            continue
        qpath = bsp_pool_en() / "truncated_queries" / level / f"{uid}.txt"
        if not qpath.exists():
            raise SystemExit(f"Missing truncated query: {qpath}")
        text = qpath.read_text(encoding="utf-8")
        row: dict = {"user_id": uid, "T_level": level, "status": "error", "error": ""}
        try:
            result = extract_profile(
                text,
                client=client,
                model=extract_model,
                extract_template=extract_tpl,
                merge_template=merge_tpl,
                max_tokens=max_tokens,
                enable_thinking=False,
                seed=None,
            )
            (extract_dir / f"{uid}.txt").write_text(result.summary + "\n", encoding="utf-8")
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
                    "n_words": len(text.split()),
                }
            )
            print(f"extract ok {uid} chunked={result.chunked} n_chunks={result.n_chunks}")
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            row["status"] = "permanent_error"
            row["error_type"] = type(exc).__name__
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"extract PERMANENT {uid}: {row['error']}")
        except Exception as exc:  # noqa: BLE001
            row["error_type"] = type(exc).__name__
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"extract ERR {uid}: {row['error']}")

        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row.get("status") == "ok":
            meta[uid] = row

    return load_last_ok_by_user(meta_path)


def phase_search(
    users: list[str],
    *,
    extract_dir: Path,
    out_dir: Path,
) -> Path:
    cand_ids, cand_matrix = load_candidate_index()
    summaries: list[str] = []
    ok_users: list[str] = []
    for uid in users:
        sp = extract_dir / f"{uid}.txt"
        if not sp.exists():
            print(f"WARN skip search {uid}: no summary")
            continue
        summaries.append(sp.read_text(encoding="utf-8").strip())
        ok_users.append(uid)

    if not ok_users:
        raise SystemExit("No summaries for search phase")

    q_vecs = embed_texts(summaries)
    hits = search_top_k(q_vecs, cand_matrix, cand_ids, k=15)

    search_path = out_dir / "search_top15.csv"
    with search_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["query_user_id", "rank", "candidate_user_id", "score"],
        )
        w.writeheader()
        for uid, user_hits in zip(ok_users, hits):
            for h in user_hits:
                w.writerow(
                    {
                        "query_user_id": uid,
                        "rank": h.rank,
                        "candidate_user_id": h.candidate_user_id,
                        "score": f"{h.score:.6f}",
                    }
                )
    print(f"search wrote {search_path} ({len(ok_users)} queries)")
    return search_path


def phase_reason(
    users: list[str],
    *,
    extract_dir: Path,
    search_path: Path,
    client,
    reason_model: str,
    reason_tpl: str,
    out_dir: Path,
    seed: int,
    resume: bool,
    max_tokens: int,
    fixture_id: str,
    level: str,
) -> Path:
    cand_summ_dir = bsp_pool_en() / "candidate_summaries"
    by_query: dict[str, list[dict]] = {u: [] for u in users}
    with search_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_query[row["query_user_id"]].append(row)

    out_jsonl = out_dir / "reason_predictions.jsonl"
    done = load_resume_skip_user_ids(out_jsonl) if resume else set()

    for uid in users:
        if uid in done:
            continue
        rows = sorted(by_query.get(uid, []), key=lambda r: int(r["rank"]))
        if len(rows) < 15:
            print(f"WARN {uid}: only {len(rows)} search hits")
        qsum_path = extract_dir / f"{uid}.txt"
        if not qsum_path.exists():
            continue
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
        row: dict = {
            "fixture_id": fixture_id,
            "T_level": level,
            "query_user_id": uid,
            "status": "error",
            "error": "",
            "error_type": "",
            "model": reason_model,
            "max_tokens": max_tokens,
        }
        try:
            result = generate(
                [{"role": "user", "content": user_prompt}],
                model=reason_model,
                client=client,
                temperature=0.0,
                max_tokens=max_tokens,
                seed=seed,
                enable_thinking=False,
            )
            obj = extract_json_object(result.text)
            pred_id, pred_num, err = resolve_predicted_candidate_id(
                obj, [c["candidate_user_id"] for c in cands]
            )
            if err:
                raise ValueError(err)
            row.update(
                {
                    "status": "ok",
                    "selected_candidate_user_id": pred_id,
                    "selected_candidate_number": pred_num,
                    "correct": pred_id == uid,
                    "true_in_top15": any(c["candidate_user_id"] == uid for c in cands),
                }
            )
            print(f"reason ok {uid} pick={pred_num} correct={row['correct']}")
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            row["status"] = "permanent_error"
            row["error_type"] = type(exc).__name__
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"reason PERMANENT {uid}: {row['error']}")
        except Exception as exc:  # noqa: BLE001
            row["error_type"] = type(exc).__name__
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"reason ERR {uid}: {row['error']}")

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    return out_jsonl


def evaluate(
    *,
    reason_jsonl: Path,
    extract_meta: dict[str, dict],
    fixture_id: str,
) -> dict:
    # Prefer last ok row per query_user_id so --resume retries that append
    # after prior error (or duplicate ok) lines do not inflate metrics.
    last_ok = load_last_ok_by_user(reason_jsonl)
    ok = list(last_ok.values())
    n = len(ok)
    top1 = sum(1 for p in ok if p.get("correct"))
    hit15 = sum(1 for p in ok if p.get("true_in_top15"))
    chunked_users = sorted(uid for uid, m in extract_meta.items() if m.get("chunked"))
    single_users = sorted(uid for uid, m in extract_meta.items() if not m.get("chunked"))
    single_set, chunked_set = set(single_users), set(chunked_users)
    single_ok = [p for p in ok if p["query_user_id"] in single_set]
    chunked_ok = [p for p in ok if p["query_user_id"] in chunked_set]

    manifest_path = fixture_dir(fixture_id) / "manifest.json"
    ref = json.loads(manifest_path.read_text(encoding="utf-8"))["bsp_reference_baselines_T8"]

    metrics = {
        "n_reason_ok": n,
        "top1_correct": top1,
        "top1_accuracy": top1 / n if n else 0.0,
        "hit_at_15_count": hit15,
        "hit_at_15": hit15 / n if n else 0.0,
        "n_chunked_extract": len(chunked_users),
        "n_single_extract": len(single_users),
        "chunked_user_ids": chunked_users,
        "single_top1": sum(1 for p in single_ok if p.get("correct")),
        "single_hit15": sum(1 for p in single_ok if p.get("true_in_top15")),
        "chunked_top1": sum(1 for p in chunked_ok if p.get("correct")),
        "chunked_hit15": sum(1 for p in chunked_ok if p.get("true_in_top15")),
        "fixture_reference_top1": ref["top1_correct_count"],
        "fixture_reference_hit15": ref["hit_at_15_count"],
        "note": (
            "Fixture baselines used BSP gpt-4o summaries+FAISS; this run uses local Extract+Search. "
            "Chunking should not break short profiles; compare pipeline completion and chunk flags."
        ),
    }
    return metrics


def main() -> int:
    load_dotenv_files(summer_root())
    args = parse_args()
    users = fixture_user_ids(args.fixture_id)
    if args.limit is not None:
        users = users[: args.limit]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else summer_root()
        / "results"
        / "runs"
        / f"p1_baseline_{args.fixture_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_tpl = load_prompt_template(bsp_prompt_summarization())
    merge_tpl = load_prompt_template(summer_prompt_extract_merge())
    reason_tpl = load_reason_template(bsp_prompt_record_selection())
    client = get_client()

    extract_meta: dict[str, dict] = {}
    if args.phase in ("all", "extract"):
        extract_meta = phase_extract(
            users,
            level=args.level,
            client=client,
            extract_model=args.extract_model,
            extract_tpl=extract_tpl,
            merge_tpl=merge_tpl,
            out_dir=out_dir,
            resume=args.resume,
            max_tokens=args.extract_max_tokens,
        )
    elif (out_dir / "extract_meta.jsonl").exists():
        extract_meta = load_last_ok_by_user(out_dir / "extract_meta.jsonl")

    search_path = out_dir / "search_top15.csv"
    if args.phase in ("all", "search"):
        phase_search(users, extract_dir=out_dir / "extract", out_dir=out_dir)

    reason_jsonl = out_dir / "reason_predictions.jsonl"
    if args.phase in ("all", "reason"):
        if not search_path.exists():
            raise SystemExit(f"Missing {search_path}; run --phase search first")
        phase_reason(
            users,
            extract_dir=out_dir / "extract",
            search_path=search_path,
            client=client,
            reason_model=args.reason_model,
            reason_tpl=reason_tpl,
            out_dir=out_dir,
            seed=args.seed,
            resume=args.resume,
            max_tokens=args.reason_max_tokens,
            fixture_id=args.fixture_id,
            level=args.level,
        )

    metrics = {}
    if reason_jsonl.exists():
        metrics = evaluate(
            reason_jsonl=reason_jsonl,
            extract_meta=extract_meta,
            fixture_id=args.fixture_id,
        )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(
            f"\nMetrics: top1={metrics['top1_correct']}/{metrics['n_reason_ok']} "
            f"hit@15={metrics['hit_at_15_count']}/{metrics['n_reason_ok']} "
            f"chunked={metrics['n_chunked_extract']}"
        )
        print(
            f"Fixture ref (BSP gpt-4o): top1={metrics['fixture_reference_top1']}/50 "
            f"hit@15={metrics['fixture_reference_hit15']}/50"
        )

    write_manifest(
        out_dir / "manifest.json",
        {
            "task": "p1_baseline_regression_50",
            "fixture_id": args.fixture_id,
            "T_level": args.level,
            "extract_model": args.extract_model,
            "reason_model": args.reason_model,
            "phase": args.phase,
            "n_users": len(users),
            "extract_prompt": str(bsp_prompt_summarization()),
            "merge_prompt": str(summer_prompt_extract_merge()),
            "reason_prompt": str(bsp_prompt_record_selection()),
            "metrics": metrics,
            "git_commit": git_commit(summer_root()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
