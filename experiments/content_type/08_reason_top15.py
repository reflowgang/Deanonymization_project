from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REASON_MODEL, paths, type_folders

TEMPERATURE = 0
SLEEP_SECONDS = 0.2
PROMPT_REL_PATH = Path("prompts/record_selection_lermen_g2.txt")

JSON_OUTPUT_INSTRUCTIONS = """

Return ONLY valid JSON with this schema:
{
  "selected_candidate_number": 1,
  "confidence": 0.0,
  "reasoning_short": "brief explanation"
}

Rules:
- selected_candidate_number must be an integer from 1 to the number of candidates listed above.
- Choose by the [number] label only; do not invent candidate_user_id values.
- confidence must be between 0 and 1.
"""

LOG_FIELDNAMES = [
    "content_type",
    "query_user_id",
    "predicted_candidate_user_id",
    "confidence",
    "reasoning_short",
    "status",
    "error",
    "model",
    "prompt_file",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Content-type experiment Reason step (gpt-4o-mini).")
    p.add_argument("--limit", type=int, default=None, help="Limit queries processed (testing).")
    p.add_argument(
        "--type",
        choices=list(type_folders()),
        default=None,
        help="Process only one content type.",
    )
    p.add_argument("--model", type=str, default=REASON_MODEL, help="OpenAI model name.")
    return p.parse_args()


def load_api_key(project_root: Path) -> None:
    env_path = project_root / "api_key.env"
    if not env_path.exists():
        env_path = project_root / ".env"
    load_dotenv(env_path, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in {env_path}.")


def get_client(project_root: Path) -> OpenAI:
    load_api_key(project_root)
    return OpenAI()


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_user_prompt(template: str, query_summary: str, candidate_block: str) -> str:
    prompt = template
    if "{query_summary}" in prompt:
        prompt = prompt.replace("{query_summary}", query_summary)
    else:
        prompt = prompt.rstrip() + "\n\nQUERY:\n" + query_summary

    if "{candidate_block}" in prompt:
        prompt = prompt.replace("{candidate_block}", candidate_block)
    else:
        prompt = prompt.rstrip() + "\n\nCANDIDATES:\n" + candidate_block

    return prompt.rstrip() + JSON_OUTPUT_INSTRUCTIONS


def build_candidate_block(candidates: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates, start=1):
        parts.append(
            f"[{i}] candidate_user_id: {c['candidate_user_id']}\n"
            f"    rank: {c['rank']}\n"
            f"    score: {float(c['score']):.6f}\n"
            f"    summary: {c['summary']}\n"
        )
    return "\n".join(parts).strip()


def parse_candidate_number(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    m = re.match(r"^(?:candidate\s*)?(\d+)$", s, re.I)
    return int(m.group(1)) if m else None


def resolve_predicted_candidate_id(
    out_obj: dict[str, Any], candidate_ids: list[str]
) -> tuple[str, str]:
    n = len(candidate_ids)
    if n == 0:
        return "", "no candidates provided"

    num = parse_candidate_number(out_obj.get("selected_candidate_number"))
    if num is not None:
        if 1 <= num <= n:
            return candidate_ids[num - 1], ""
        return "", f"selected_candidate_number out of range: {num} (valid 1..{n})"

    raw_id = str(out_obj.get("predicted_candidate_user_id", "")).strip()
    if raw_id:
        if raw_id in candidate_ids:
            return raw_id, ""
        as_num = parse_candidate_number(raw_id)
        if as_num is not None and 1 <= as_num <= n:
            return candidate_ids[as_num - 1], ""
        return "", f"could not map prediction to candidate list: {raw_id!r}"

    return "", "missing selected_candidate_number and predicted_candidate_user_id"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def ensure_out_csv_header(out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        return
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDNAMES).writeheader()


def append_row(out_csv: Path, row: dict[str, object]) -> None:
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDNAMES).writerow(row)


def load_done_ok(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty or not {"content_type", "query_user_id", "status"}.issubset(df.columns):
        return set()
    ok_df = df[df["status"].astype(str) == "ok"]
    return {
        (str(r["content_type"]), str(r["query_user_id"])) for _, r in ok_df.iterrows()
    }


def call_openai_reason(
    client: OpenAI,
    model: str,
    prompt_template: str,
    query_summary: str,
    candidate_block: str,
) -> dict[str, Any]:
    user_content = build_user_prompt(prompt_template, query_summary, candidate_block)
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": user_content}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("Model returned non-object JSON.")
    return obj


def main() -> None:
    args = parse_args()
    p = paths()
    project_root = Path(__file__).resolve().parents[2]
    model = args.model.strip()
    prompt_file_log = str(PROMPT_REL_PATH).replace("\\", "/")

    if not p["faiss_out"].exists():
        raise FileNotFoundError(f"FAISS CSV not found: {p['faiss_out']}. Run 07_search_faiss.py first.")

    prompt_template = load_prompt_template(p["reason_prompt"])
    ensure_out_csv_header(p["reason_out"])
    done_ok = load_done_ok(p["reason_out"])

    df = pd.read_csv(p["faiss_out"])
    required = {"content_type", "query_user_id", "candidate_user_id", "rank", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FAISS CSV missing columns: {sorted(missing)}")

    if args.type:
        df = df[df["content_type"].astype(str) == args.type].copy()

    df = df.sort_values(["content_type", "query_user_id", "rank"], kind="mergesort")
    groups = list(df.groupby(["content_type", "query_user_id"], sort=False))
    if args.limit is not None:
        groups = groups[: max(0, args.limit)]

    print(f"Queries to process: {len(groups):,}")
    print(f"Model: {model}")

    client: Optional[OpenAI] = None

    for (content_type, query_user_id), g in groups:
        key = (str(content_type), str(query_user_id))
        if key in done_ok:
            continue

        status = "ok"
        error = ""
        predicted_candidate_user_id = ""
        confidence = 0.0
        reasoning_short = ""
        called_api = False

        try:
            query_path = p["type_summaries_root"] / str(content_type) / f"{query_user_id}.txt"
            if not query_path.exists():
                raise FileNotFoundError(f"Missing query summary: {query_path}")
            query_summary = read_text(query_path)
            if not query_summary:
                raise ValueError("Empty query summary.")

            candidates: list[dict[str, object]] = []
            candidate_ids: list[str] = []
            for _, row in g.iterrows():
                cid = str(row["candidate_user_id"])
                cand_path = p["candidate_summaries"] / f"{cid}.txt"
                if not cand_path.exists():
                    raise FileNotFoundError(f"Missing candidate summary: {cand_path}")
                candidates.append(
                    {
                        "candidate_user_id": cid,
                        "rank": int(row["rank"]),
                        "score": float(row["score"]),
                        "summary": read_text(cand_path),
                    }
                )
                candidate_ids.append(cid)

            if client is None:
                client = get_client(project_root)

            called_api = True
            out_obj = call_openai_reason(
                client, model, prompt_template, query_summary, build_candidate_block(candidates)
            )
            confidence = clamp01(out_obj.get("confidence", 0.0))
            reasoning_short = str(out_obj.get("reasoning_short", "")).strip()
            predicted_candidate_user_id, map_err = resolve_predicted_candidate_id(
                out_obj, candidate_ids
            )
            if map_err:
                status = "error"
                error = map_err
        except Exception as e:  # noqa: BLE001
            status = "error"
            error = str(e)

        append_row(
            p["reason_out"],
            {
                "content_type": str(content_type),
                "query_user_id": query_user_id,
                "predicted_candidate_user_id": predicted_candidate_user_id,
                "confidence": float(confidence),
                "reasoning_short": reasoning_short,
                "status": status,
                "error": error,
                "model": model,
                "prompt_file": prompt_file_log,
            },
        )
        if status == "ok":
            done_ok.add(key)
        if called_api:
            time.sleep(SLEEP_SECONDS)

    print(f"\nWrote/updated: {p['reason_out']}")


if __name__ == "__main__":
    main()
