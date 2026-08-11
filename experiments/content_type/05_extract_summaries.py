from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONTENT_TYPES, EXTRACT_MODEL, TYPE_LABELS, paths, type_folders

TEMPERATURE = 0
SLEEP_SECONDS = 0.2
PROMPT_REL_PATH = Path("prompts/summarization_lermen_g2.txt")

LOG_FIELDNAMES = [
    "content_type",
    "user_id",
    "input_path",
    "output_path",
    "status",
    "error",
    "input_words",
    "output_words",
    "model",
    "prompt_file",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract summaries for P/O/T type profiles.")
    p.add_argument(
        "--type",
        choices=list(type_folders()),
        default=None,
        help="Process only one content type folder.",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit profiles per type (testing).")
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


def build_user_prompt(template: str, profile_text: str) -> str:
    if "{profile_text}" in template:
        return template.replace("{profile_text}", profile_text)
    return template.rstrip() + "\n\nCOMMENTS:\n" + profile_text


def count_words(text: str) -> int:
    return len(text.split())


def append_log_row(log_csv: Path, row: dict[str, object]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_csv.exists()
    with log_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def call_openai_summary(client: OpenAI, model: str, prompt_template: str, profile_text: str) -> str:
    user_content = build_user_prompt(prompt_template, profile_text)
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": user_content}],
    )
    return (resp.choices[0].message.content or "").strip()


def folder_for_label(label: str) -> str:
    return TYPE_LABELS[label]


def main() -> None:
    args = parse_args()
    p = paths()
    project_root = Path(__file__).resolve().parents[2]
    prompt_file_log = str(PROMPT_REL_PATH).replace("\\", "/")
    prompt_template = load_prompt_template(p["summarize_prompt"])
    client = get_client(project_root)

    selected_folders = list(type_folders())
    if args.type:
        selected_folders = [args.type]

    for folder in selected_folders:
        (p["type_summaries_root"] / folder).mkdir(parents=True, exist_ok=True)

    print(f"Model: {EXTRACT_MODEL}")
    print(f"Profiles root: {p['type_profiles_root']}")
    print(f"Summaries root: {p['type_summaries_root']}")

    for label in CONTENT_TYPES:
        folder = folder_for_label(label)
        if folder not in selected_folders:
            continue

        in_dir = p["type_profiles_root"] / folder
        out_dir = p["type_summaries_root"] / folder
        if not in_dir.exists():
            print(f"- {folder}: no profiles directory, skipping")
            continue

        files = sorted(in_dir.glob("*.txt"))
        if args.limit is not None:
            files = files[: max(0, args.limit)]

        print(f"- {folder}: {len(files):,} profile files")

        for in_path in files:
            user_id = in_path.stem
            out_path = out_dir / f"{user_id}.txt"
            base = {
                "content_type": folder,
                "user_id": user_id,
                "input_path": str(in_path),
                "output_path": str(out_path),
                "model": EXTRACT_MODEL,
                "prompt_file": prompt_file_log,
            }

            if out_path.exists() and out_path.stat().st_size > 0:
                profile_text = in_path.read_text(encoding="utf-8")
                summary_text = out_path.read_text(encoding="utf-8")
                append_log_row(
                    p["extract_log"],
                    {
                        **base,
                        "status": "skipped_existing",
                        "error": "",
                        "input_words": count_words(profile_text),
                        "output_words": count_words(summary_text),
                    },
                )
                continue

            try:
                profile_text = in_path.read_text(encoding="utf-8")
                input_words = count_words(profile_text)
                if input_words == 0:
                    append_log_row(
                        p["extract_log"],
                        {
                            **base,
                            "status": "error",
                            "error": "empty_input_profile",
                            "input_words": 0,
                            "output_words": 0,
                        },
                    )
                    continue

                summary_text = call_openai_summary(
                    client, EXTRACT_MODEL, prompt_template, profile_text
                )
                out_path.write_text(summary_text + ("\n" if summary_text else ""), encoding="utf-8")
                append_log_row(
                    p["extract_log"],
                    {
                        **base,
                        "status": "ok",
                        "error": "",
                        "input_words": input_words,
                        "output_words": count_words(summary_text),
                    },
                )
                time.sleep(SLEEP_SECONDS)
            except Exception as e:  # noqa: BLE001
                append_log_row(
                    p["extract_log"],
                    {
                        **base,
                        "status": "error",
                        "error": str(e),
                        "input_words": count_words(in_path.read_text(encoding="utf-8"))
                        if in_path.exists()
                        else 0,
                        "output_words": 0,
                    },
                )
                time.sleep(SLEEP_SECONDS)

    print(f"\nWrote/updated log: {p['extract_log']}")
    print("Done.")


if __name__ == "__main__":
    main()
