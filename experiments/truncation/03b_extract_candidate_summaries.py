from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4o-mini"
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
MAX_INPUT_WORDS = 25000
PROMPT_REL_PATH = Path("prompts/summarization_lermen_g2.txt")


@dataclass(frozen=True)
class Paths:
    candidate_root: Path
    extracted_candidate_root: Path
    extract_log_csv: Path
    prompt_file: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract candidate summaries from HN JSONL using the Lermen G.2 prompt."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of users processed (for testing).",
    )
    return p.parse_args(argv)


def load_api_key_from_env(project_root: Path) -> None:
    env_path = project_root / "api_key.env"
    if not env_path.exists():
        env_path = project_root / ".env"

    load_dotenv(env_path, override=False)
    print(f"Loaded env file: {env_path}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in {env_path}.")


def get_client(project_root: Path) -> OpenAI:
    load_api_key_from_env(project_root)
    return OpenAI()


def load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_user_prompt(template: str, profile_text: str) -> str:
    if "{profile_text}" in template:
        return template.replace("{profile_text}", profile_text)
    return template.rstrip() + "\n\nCOMMENTS:\n" + profile_text


def list_candidate_files(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists():
        return []
    return sorted(p for p in candidate_dir.glob("user_*_candidate.jsonl") if p.is_file())


def user_id_from_candidate_path(path: Path) -> str:
    name = path.name
    return name.split("_candidate.jsonl")[0]


def read_candidate_text(candidate_jsonl_path: Path) -> str:
    texts: list[str] = []
    with candidate_jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {candidate_jsonl_path}: {e}"
                ) from e
            if "text" not in obj:
                raise ValueError(
                    f'Missing "text" field on line {line_no} in {candidate_jsonl_path}'
                )
            texts.append(str(obj["text"]))
    return " ".join(texts).strip()


def count_words(text: str) -> int:
    return len(text.split())


def truncate_to_first_n_words(text: str, max_words: int) -> tuple[str, int, bool]:
    words = text.split()
    original_words = len(words)
    if original_words <= max_words:
        return text, original_words, False
    truncated = " ".join(words[:max_words])
    return truncated, original_words, True


def call_openai_summary(client: OpenAI, prompt_template: str, profile_text: str) -> str:
    user_content = build_user_prompt(prompt_template, profile_text)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": user_content}],
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def summary_text_to_json_obj(summary_text: str) -> dict[str, Any]:
    return {"summary": summary_text}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    limit = args.limit

    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        candidate_root=project_root / "data/splits/candidate",
        extracted_candidate_root=project_root / "data/extracted_summaries/candidate",
        extract_log_csv=project_root / "results/tables/hn_candidate_extract_log.csv",
        prompt_file=project_root / PROMPT_REL_PATH,
    )
    _ensure_dir(paths.extracted_candidate_root)
    _ensure_parent_dir(paths.extract_log_csv)

    prompt_template = load_prompt_template(paths.prompt_file)

    print(f"Input candidate JSONL: {paths.candidate_root}")
    print(f"Output summaries:      {paths.extracted_candidate_root}")
    print(f"Prompt file:           {paths.prompt_file}")
    print(f"Model: {MODEL} (temperature={TEMPERATURE})")
    print(f"Max input words (sent to API): {MAX_INPUT_WORDS:,}")

    candidate_files = list_candidate_files(paths.candidate_root)
    if limit is not None:
        candidate_files = candidate_files[: max(0, limit)]

    print(f"Candidate profiles to consider: {len(candidate_files):,}")

    log_rows: list[dict[str, Any]] = []
    client: Optional[OpenAI] = None

    for cand_path in candidate_files:
        user_id = user_id_from_candidate_path(cand_path)
        out_path = paths.extracted_candidate_root / f"{user_id}_summary.json"

        if out_path.exists():
            log_rows.append(
                {
                    "user_id": user_id,
                    "input_path": str(cand_path),
                    "output_path": str(out_path),
                    "input_words": None,
                    "status": "skipped_exists",
                    "error": "",
                }
            )
            continue

        try:
            profile_text = read_candidate_text(cand_path)
            sent_text, original_input_words, was_truncated = truncate_to_first_n_words(
                profile_text, MAX_INPUT_WORDS
            )
            input_words = count_words(sent_text)

            if original_input_words == 0:
                log_rows.append(
                    {
                        "user_id": user_id,
                        "input_path": str(cand_path),
                        "output_path": str(out_path),
                        "input_words": 0,
                        "status": "skipped_empty",
                        "error": "",
                    }
                )
                continue

            if client is None:
                client = get_client(project_root)

            summary_text = call_openai_summary(client, prompt_template, sent_text)
            write_json(out_path, summary_text_to_json_obj(summary_text))

            log_rows.append(
                {
                    "user_id": user_id,
                    "input_path": str(cand_path),
                    "output_path": str(out_path),
                    "input_words": int(input_words),
                    "status": "success",
                    "error": "",
                }
            )
            time.sleep(SLEEP_SECONDS)
        except Exception as e:  # noqa: BLE001
            log_rows.append(
                {
                    "user_id": user_id,
                    "input_path": str(cand_path),
                    "output_path": str(out_path),
                    "input_words": None,
                    "status": "error",
                    "error": str(e),
                }
            )
            print(f"  ! {user_id}: error ({e})")
            time.sleep(SLEEP_SECONDS)

    pd.DataFrame(log_rows).to_csv(paths.extract_log_csv, index=False)
    print(f"\nWrote run log: {paths.extract_log_csv} ({len(log_rows):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
