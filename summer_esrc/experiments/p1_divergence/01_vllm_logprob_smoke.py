#!/usr/bin/env python3
"""P1.1 — vLLM log-probability smoke test (no pool data).

Verifies:
  1) VLLM_BASE_URL is reachable
  2) /v1/models returns at least one model
  3) chat.completions with logprobs=True returns non-empty token logprobs
  4) a run manifest is written under results/runs/

Usage (on the university server, with vLLM already serving):

  export VLLM_BASE_URL=http://127.0.0.1:8000/v1
  # optional: export VLLM_SMOKE_MODEL=...
  python experiments/p1_divergence/01_vllm_logprob_smoke.py

Or from project root with a .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from esrc.config import load_dotenv_files, project_root  # noqa: E402
from esrc.generate import generate, get_client, list_models  # noqa: E402
from esrc.manifests import git_commit, sha256_text, write_manifest  # noqa: E402

SMOKE_PROMPT = "Reply with a single lowercase word: yes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1.1 vLLM logprob smoke test")
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override VLLM_BASE_URL (e.g. http://127.0.0.1:8000/v1)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id (default: VLLM_SMOKE_MODEL or first /v1/models entry)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="Request top-k logprobs per token (default: 5)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override results/runs/<run_id> directory",
    )
    return p.parse_args()


def main() -> int:
    load_dotenv_files(project_root())
    args = parse_args()

    if args.base_url:
        os.environ["VLLM_BASE_URL"] = args.base_url.rstrip("/")

    print("=== P1.1 vLLM logprob smoke test ===")
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        print(
            "FAIL: VLLM_BASE_URL is unset.\n"
            "  Set it to your OpenAI-compatible vLLM endpoint, e.g.\n"
            "  export VLLM_BASE_URL=http://127.0.0.1:8000/v1"
        )
        return 2

    print(f"base_url: {base_url}")

    try:
        client = get_client(base_url=base_url)
        models = list_models(client)
    except Exception as exc:  # noqa: BLE001 — smoke test must surface any connection error
        print(f"FAIL: could not list models from {base_url}: {exc}")
        return 1

    if not models:
        print("FAIL: /v1/models returned an empty list")
        return 1

    print(f"models ({len(models)}): {models[:8]}{'...' if len(models) > 8 else ''}")

    model = args.model or os.getenv("VLLM_SMOKE_MODEL") or models[0]
    print(f"smoke model: {model}")
    print(f"prompt: {SMOKE_PROMPT!r}")

    try:
        result = generate(
            [{"role": "user", "content": SMOKE_PROMPT}],
            model=model,
            client=client,
            temperature=0.0,
            max_tokens=8,
            logprobs=True,
            top_logprobs=args.top_logprobs,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: generate() with logprobs=True raised: {exc}")
        return 1

    print(f"response text: {result.text!r}")
    print(f"finish_reason: {result.finish_reason}")
    print(f"n token_logprobs: {len(result.token_logprobs)}")
    print(f"sequence_logprob: {result.sequence_logprob}")

    if not result.token_logprobs:
        print(
            "FAIL: logprobs requested but token_logprobs is empty.\n"
            "  Check that vLLM was started with logprobs enabled and that the\n"
            "  OpenAI-compatible chat endpoint returns choice.logprobs.content."
        )
        return 1

    preview = result.token_logprobs[:5]
    for i, t in enumerate(preview):
        print(f"  token[{i}]: {t.token!r}  logprob={t.logprob:.6f}  top_k={len(t.top_logprobs)}")

    run_id = datetime.now(timezone.utc).strftime("p1_1_smoke_%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction = {
        "prompt": SMOKE_PROMPT,
        "text": result.text,
        "model": result.model,
        "finish_reason": result.finish_reason,
        "sequence_logprob": result.sequence_logprob,
        "n_token_logprobs": len(result.token_logprobs),
        "token_logprobs_preview": [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top_logprobs": [{"token": tt, "logprob": lp} for tt, lp in t.top_logprobs[:3]],
            }
            for t in result.token_logprobs[:10]
        ],
    }
    pred_path = out_dir / "smoke_prediction.json"
    pred_path.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "task": "p1.1_vllm_logprob_smoke",
        "pool_id": "none",
        "seed": args.seed,
        "model": result.model,
        "model_requested": model,
        "base_url": base_url,
        "prompt_hash": sha256_text(SMOKE_PROMPT),
        "input_checksum": sha256_text(SMOKE_PROMPT),
        "git_commit": git_commit(ROOT),
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "status": "ok",
        "n_token_logprobs": len(result.token_logprobs),
        "sequence_logprob": result.sequence_logprob,
        "prediction_file": str(pred_path.relative_to(ROOT)),
    }
    man_path = write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote: {pred_path}")
    print(f"wrote: {man_path}")
    print("PASS: vLLM logprobs are available via generate().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
