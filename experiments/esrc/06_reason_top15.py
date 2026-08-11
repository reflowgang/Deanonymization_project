from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
PROMPT_REL_PATH = Path("prompts/record_selection_lermen_g2.txt")

LEVELS_ALL: list[str] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

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
    "T_level",
    "query_user_id",
    "predicted_candidate_user_id",
    "confidence",
    "reasoning_short",
    "status",
    "error",
    "model",
    "prompt_file",
]


@dataclass(frozen=True)
class Paths:
    faiss_csv: Path
    query_summaries_root: Path
    candidate_summaries_dir: Path
    out_csv: Path
    prompt_file: Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POOL-EN Reason step over FAISS top-15 results.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of (T_level, query_user_id) groups processed (for testing).",
    )
    p.add_argument(
        "--query_csv",
        type=str,
        default=None,
        help="Optional CSV with columns T_level, query_user_id to process only those groups.",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=LEVELS_ALL,
        help="Truncation levels to process, e.g. --levels T1 T8 (default: all).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name (default: gpt-4o-mini). Example: --model gpt-4o",
    )
    return p.parse_args(argv)


def normalize_levels(levels: list[str]) -> list[str]:
    levels_norm = [lvl.strip() for lvl in levels if lvl and lvl.strip()]
    unknown = sorted(set(levels_norm) - set(LEVELS_ALL))
    if unknown:
        raise ValueError(f"Unknown levels: {unknown}. Allowed: {LEVELS_ALL}")
    out: list[str] = []
    seen: set[str] = set()
    for lvl in levels_norm:
        if lvl not in seen:
            out.append(lvl)
            seen.add(lvl)
    return out


def load_api_key_from_env(project_root: Path) -> None:
    env_path = project_root / "api_key.env"
    if not env_path.exists():
        env_path = project_root / ".env"
    load_dotenv(env_path, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in {env_path}.")


def get_client(project_root: Path) -> OpenAI:
    load_api_key_from_env(project_root)
    return OpenAI()


def load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
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
        cid = str(c["candidate_user_id"])
        rank = int(c["rank"])
        score = float(c["score"])
        summary = str(c["summary"])
        parts.append(
            f"[{i}] candidate_user_id: {cid}\n"
            f"    rank: {rank}\n"
            f"    score: {score:.6f}\n"
            f"    summary: {summary}\n"
        )
    return "\n".join(parts).strip()


def parse_candidate_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.match(r"^(?:candidate\s*)?(\d+)$", s, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^user_(\d+)$", s, re.I)
    if m:
        return int(m.group(1))
    return None


def resolve_predicted_candidate_id(
    out_obj: dict[str, Any],
    candidate_ids: list[str],
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


def try_tqdm(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def ensure_out_csv_header(out_csv: Path) -> None:
    _ensure_parent_dir(out_csv)
    if out_csv.exists():
        return
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        w.writeheader()


def append_row(out_csv: Path, row: dict[str, object]) -> None:
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        w.writerow(row)


def load_done_ok(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    if not {"T_level", "query_user_id", "status"}.issubset(df.columns):
        return set()
    ok_df = df[df["status"].astype(str) == "ok"]
    return {
        (str(r["T_level"]), str(r["query_user_id"])) for _, r in ok_df.iterrows()
    }


def parse_model_json(content: str) -> dict[str, Any]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Model returned non-object JSON.")
    return obj


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
    return parse_model_json(content)


def load_query_csv_groups(query_csv_path: Path) -> list[tuple[str, str]]:
    if not query_csv_path.exists():
        raise FileNotFoundError(f"--query_csv file not found: {query_csv_path}")
    df = pd.read_csv(query_csv_path)
    required = {"T_level", "query_user_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"--query_csv missing columns: {sorted(missing)}")
    df["T_level"] = df["T_level"].astype(str)
    df["query_user_id"] = df["query_user_id"].astype(str)
    groups: list[tuple[str, str]] = []
    for _, r in df.iterrows():
        groups.append((str(r["T_level"]), str(r["query_user_id"])))
    return groups


def main() -> None:
    args = parse_args()
    levels = normalize_levels(list(args.levels))
    limit = args.limit
    model = args.model.strip()
    query_csv = args.query_csv.strip() if isinstance(args.query_csv, str) else None

    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        faiss_csv=project_root / "results/tables/pool_en_faiss_top15.csv",
        query_summaries_root=project_root / "data/esrc/pool_en/summaries",
        candidate_summaries_dir=project_root / "data/esrc/pool_en/candidate_summaries",
        out_csv=project_root / "results/tables/pool_en_reason_predictions.csv",
        prompt_file=project_root / PROMPT_REL_PATH,
    )
    prompt_file_log = str(PROMPT_REL_PATH).replace("\\", "/")

    if not paths.faiss_csv.exists():
        raise FileNotFoundError(f"FAISS top-15 CSV not found: {paths.faiss_csv}")

    prompt_template = load_prompt_template(paths.prompt_file)

    ensure_out_csv_header(paths.out_csv)
    done_ok = load_done_ok(paths.out_csv)

    df = pd.read_csv(paths.faiss_csv)
    required = {"T_level", "query_user_id", "candidate_user_id", "rank", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FAISS CSV missing columns: {sorted(missing)}")

    df["T_level"] = df["T_level"].astype(str)
    df["query_user_id"] = df["query_user_id"].astype(str)
    df["candidate_user_id"] = df["candidate_user_id"].astype(str)
    df = df[df["T_level"].isin(levels)].copy()

    if df.empty:
        print("No FAISS rows for requested levels. Nothing to do.")
        return

    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    df = df.sort_values(["T_level", "query_user_id", "rank"], kind="mergesort")

    if query_csv:
        query_filter_df = pd.read_csv(query_csv)
        query_filter_df["T_level"] = query_filter_df["T_level"].astype(str)
        query_filter_df["query_user_id"] = query_filter_df["query_user_id"].astype(str)
        allowed_pairs = set(zip(query_filter_df["T_level"], query_filter_df["query_user_id"]))
        df = df[df.apply(lambda r: (r["T_level"], r["query_user_id"]) in allowed_pairs, axis=1)]
        groups = list(df.groupby(["T_level", "query_user_id"], sort=False))
    else:
        groups = list(df.groupby(["T_level", "query_user_id"], sort=False))
        if limit is not None:
            groups = groups[: max(0, limit)]

    print(f"Groups to process: {len(groups):,}")
    print(f"Model: {model} (temperature={TEMPERATURE})")
    print(f"Prompt file: {paths.prompt_file}")

    client: Optional[OpenAI] = None

    for (level, query_user_id), g in try_tqdm(groups, total=len(groups), desc="reason"):
        key = (str(level), str(query_user_id))
        if key in done_ok:
            continue

        status = "ok"
        error = ""
        predicted_candidate_user_id = ""
        confidence = 0.0
        reasoning_short = ""
        called_api = False

        try:
            query_path = paths.query_summaries_root / str(level) / f"{query_user_id}.txt"
            if not query_path.exists():
                raise FileNotFoundError(f"Missing query summary: {query_path}")
            query_summary = read_text(query_path)
            if not query_summary:
                raise ValueError("Empty query summary text.")

            candidates: list[dict[str, object]] = []
            candidate_ids: list[str] = []
            for _, row in g.iterrows():
                cid = str(row["candidate_user_id"])
                cand_path = paths.candidate_summaries_dir / f"{cid}.txt"
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

            candidate_block = build_candidate_block(candidates)

            if client is None:
                client = get_client(project_root)

            called_api = True
            out_obj = call_openai_reason(
                client=client,
                model=model,
                prompt_template=prompt_template,
                query_summary=query_summary,
                candidate_block=candidate_block,
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
            paths.out_csv,
            {
                "T_level": str(level),
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

    print(f"\nWrote/updated: {paths.out_csv}")


if __name__ == "__main__":
    main()
