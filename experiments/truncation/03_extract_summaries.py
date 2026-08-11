from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4o-mini"
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
PROMPT_REL_PATH = Path("prompts/summarization_lermen_g2.txt")
LEVELS_ALL: list[str] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]


@dataclass(frozen=True)
class Paths:
    summaries_root: Path
    extracted_root: Path
    extract_log_csv: Path
    prompt_file: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract summaries from HN user profiles using the Lermen G.2 prompt."
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=LEVELS_ALL,
        help="Levels to process, e.g. --levels T1 T2 (default: all).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of users per level (for testing).",
    )
    return p.parse_args(argv)


def normalize_levels(levels: Iterable[str]) -> list[str]:
    levels_norm: list[str] = []
    for lvl in levels:
        lvl = lvl.strip()
        if not lvl:
            continue
        levels_norm.append(lvl)
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


def list_user_profiles(level_dir: Path) -> list[Path]:
    if not level_dir.exists():
        return []
    return sorted(p for p in level_dir.glob("user_*.txt") if p.is_file())


def user_id_from_profile_path(path: Path) -> str:
    return path.stem


def count_words(text: str) -> int:
    return len(text.split())


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


def build_user_prompt(template: str, profile_text: str) -> str:
    if "{profile_text}" in template:
        return template.replace("{profile_text}", profile_text)
    return template.rstrip() + "\n\nCOMMENTS:\n" + profile_text


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
    # Downstream HN scripts expect JSON files; Lermen extract returns plain text.
    return {"summary": summary_text}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    levels = normalize_levels(args.levels)
    limit = args.limit

    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        summaries_root=project_root / "data/summaries",
        extracted_root=project_root / "data/extracted_summaries",
        extract_log_csv=project_root / "results/tables/hn_extract_log.csv",
        prompt_file=project_root / PROMPT_REL_PATH,
    )
    _ensure_parent_dir(paths.extract_log_csv)
    for lvl in LEVELS_ALL:
        _ensure_dir(paths.extracted_root / lvl)

    prompt_template = load_prompt_template(paths.prompt_file)

    print(f"Input profiles:  {paths.summaries_root}")
    print(f"Output summaries:{paths.extracted_root}")
    print(f"Prompt file:     {paths.prompt_file}")
    print(f"Model: {MODEL} (temperature={TEMPERATURE})")

    log_rows: list[dict[str, Any]] = []
    client: Optional[OpenAI] = None

    for level in levels:
        in_dir = paths.summaries_root / level
        out_dir = paths.extracted_root / level

        profiles = list_user_profiles(in_dir)
        if not profiles:
            print(f"- {level}: no users found (skipping level)")
            continue

        if limit is not None:
            profiles = profiles[: max(0, limit)]

        print(f"- {level}: {len(profiles):,} user profiles to consider")

        for profile_path in profiles:
            user_id = user_id_from_profile_path(profile_path)
            out_path = out_dir / f"{user_id}_summary.json"

            if out_path.exists():
                log_rows.append(
                    {
                        "level": level,
                        "user_id": user_id,
                        "input_path": str(profile_path),
                        "output_path": str(out_path),
                        "input_words": None,
                        "status": "skipped_exists",
                        "error": "",
                    }
                )
                continue

            text = profile_path.read_text(encoding="utf-8")
            input_words = count_words(text)

            if input_words == 0:
                log_rows.append(
                    {
                        "level": level,
                        "user_id": user_id,
                        "input_path": str(profile_path),
                        "output_path": str(out_path),
                        "input_words": int(input_words),
                        "status": "skipped_empty",
                        "error": "",
                    }
                )
                continue

            try:
                if client is None:
                    client = get_client(project_root)

                summary_text = call_openai_summary(client, prompt_template, text)
                write_json(out_path, summary_text_to_json_obj(summary_text))

                log_rows.append(
                    {
                        "level": level,
                        "user_id": user_id,
                        "input_path": str(profile_path),
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
                        "level": level,
                        "user_id": user_id,
                        "input_path": str(profile_path),
                        "output_path": str(out_path),
                        "input_words": int(input_words),
                        "status": "error",
                        "error": str(e),
                    }
                )
                print(f"  ! {level}/{user_id}: error ({e})")
                time.sleep(SLEEP_SECONDS)

    pd.DataFrame(log_rows).to_csv(paths.extract_log_csv, index=False)
    print(f"\nWrote run log: {paths.extract_log_csv} ({len(log_rows):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
