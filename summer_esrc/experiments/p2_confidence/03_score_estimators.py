#!/usr/bin/env python3
"""P2.4–P2.7 — Compute estimators (a)/(b)/(c) from Reason picks + per-candidate scores.

(c) scores each of 15 candidates with a forced-hypothesis call (thinking off),
softmaxes, and takes mass on the P2.3-selected index (same pick).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from esrc.confidence import (  # noqa: E402
    estimator_a_verbalized,
    estimator_b_selected_id_logprob,
    estimator_c_softmax_mass_on_selected,
    logprob_to_unit_interval,
)
from esrc.config import load_dotenv_files  # noqa: E402
from esrc.generate import TokenLogprob, generate, get_client  # noqa: E402
from esrc.manifests import git_commit, write_manifest  # noqa: E402
from esrc.paths import bsp_prompt_record_selection, summer_root  # noqa: E402
from esrc.reason_prompt import (  # noqa: E402
    build_candidate_block,
    build_forced_candidate_prompt,
    load_prompt_template,
)

DEFAULT_REASON_MODEL = "qwen3.6-35b-a3b-nvfp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 estimators a/b/c")
    p.add_argument("--reason-jsonl", type=str, required=True)
    p.add_argument(
        "--inputs-dir",
        type=str,
        default=str(summer_root() / "results/runs/p2_inputs_regression_50_T8/inputs"),
    )
    p.add_argument("--model", type=str, default=os.getenv("VLLM_REASON_MODEL", DEFAULT_REASON_MODEL))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--skip-c", action="store_true", help="Only compute a/b (no 15× calls)")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip query_user_ids already status=ok in estimator_scores.csv",
    )
    return p.parse_args()


def load_done_ok(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and r.get("query_user_id"):
                done.add(r["query_user_id"])
    return done


def load_reason_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens_from_row(row: dict) -> list[TokenLogprob]:
    raw = row.get("token_logprobs") or []
    return [TokenLogprob(token=t["token"], logprob=float(t["logprob"])) for t in raw]


def score_forced_candidate(
    *,
    client,
    model: str,
    template: str,
    pack: dict,
    candidate_number: int,
    max_tokens: int,
    seed: int,
) -> float:
    block = build_candidate_block(pack["candidates"])
    prompt = build_forced_candidate_prompt(
        template, pack["query_summary"], block, candidate_number
    )
    result = generate(
        [{"role": "user", "content": prompt}],
        model=model,
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=1,
        seed=seed,
        enable_thinking=False,
    )
    # Prefer logprob mass on the forced number tokens; fallback to full seq logprob.
    lp = estimator_b_selected_id_logprob(result.token_logprobs, candidate_number)
    if lp is None:
        if result.sequence_logprob is None:
            raise ValueError("No logprobs for forced candidate score")
        lp = float(result.sequence_logprob)
    return float(lp)


def main() -> int:
    load_dotenv_files(summer_root())
    args = parse_args()
    reason_path = Path(args.reason_jsonl)
    inputs_dir = Path(args.inputs_dir)
    rows = [r for r in load_reason_rows(reason_path) if r.get("status") == "ok"]
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else reason_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "estimator_scores.csv"

    template = load_prompt_template(bsp_prompt_record_selection())
    client = None if args.skip_c else get_client()

    fieldnames = [
        "query_user_id",
        "correct",
        "selected_candidate_number",
        "score_a",
        "score_b_logprob",
        "score_b",
        "score_c",
        "score_c_argmax_number",
        "c_argmax_disagrees_with_pick",
        "candidate_scores_json",
        "status",
        "error",
    ]
    done = load_done_ok(out_csv) if args.resume else set()
    mode = "a" if (args.resume and out_csv.exists()) else "w"
    print(f"estimator rows to consider={len(rows)} already_done={len(done)} mode={mode}")

    with out_csv.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()

        for row in rows:
            uid = row["query_user_id"]
            if uid in done:
                continue
            pack_path = inputs_dir / f"{uid}.json"
            out = {
                "query_user_id": uid,
                "correct": bool(row.get("correct")),
                "selected_candidate_number": row.get("selected_candidate_number"),
                "score_a": "",
                "score_b_logprob": "",
                "score_b": "",
                "score_c": "",
                "score_c_argmax_number": "",
                "c_argmax_disagrees_with_pick": "",
                "candidate_scores_json": "",
                "status": "error",
                "error": "",
            }
            try:
                score_a = estimator_a_verbalized(row["verbalized_confidence"])
                tokens = tokens_from_row(row)
                sel_num = int(row["selected_candidate_number"])
                score_b_lp = row.get("selected_id_logprob")
                if score_b_lp is None:
                    score_b_lp = estimator_b_selected_id_logprob(tokens, sel_num)
                if score_b_lp is None:
                    raise ValueError("Could not locate selected-number token span for (b)")
                score_b_lp = float(score_b_lp)
                score_b = logprob_to_unit_interval(score_b_lp)

                out.update(
                    {
                        "score_a": f"{score_a:.6f}",
                        "score_b_logprob": f"{score_b_lp:.6f}",
                        "score_b": f"{score_b:.6f}",
                    }
                )

                if not args.skip_c:
                    pack = json.loads(pack_path.read_text(encoding="utf-8"))
                    n_cands = len(pack["candidates"])
                    scores = []
                    for i in range(1, n_cands + 1):
                        lp = score_forced_candidate(
                            client=client,
                            model=args.model,
                            template=template,
                            pack=pack,
                            candidate_number=i,
                            max_tokens=args.max_tokens,
                            seed=args.seed,
                        )
                        scores.append(lp)
                        print(f"  {uid} c[{i}/{n_cands}] lp={lp:.4f}", flush=True)
                    score_c, probs = estimator_c_softmax_mass_on_selected(
                        scores, sel_num - 1, temperature=1.0
                    )
                    argmax_num = int(max(range(len(probs)), key=lambda i: probs[i])) + 1
                    disagrees = int(argmax_num != sel_num)
                    out.update(
                        {
                            "score_c": f"{score_c:.6f}",
                            "score_c_argmax_number": str(argmax_num),
                            "c_argmax_disagrees_with_pick": str(disagrees),
                            "candidate_scores_json": json.dumps(scores),
                        }
                    )

                out["status"] = "ok"
                print(
                    f"ok {uid} a={out['score_a']} b={out['score_b']} "
                    f"c={out['score_c'] or 'skipped'} "
                    f"c_disagree={out['c_argmax_disagrees_with_pick']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"
                print(f"ERR {uid}: {out['error']}", flush=True)

            w.writerow(out)
            f.flush()

    write_manifest(
        out_dir / "estimators_manifest.json",
        {
            "task": "p2.4_p2.7_estimators",
            "reason_jsonl": str(reason_path),
            "model": args.model,
            "skip_c": args.skip_c,
            "enable_thinking": False,
            "git_commit": git_commit(summer_root()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
