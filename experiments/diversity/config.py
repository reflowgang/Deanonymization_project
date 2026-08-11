from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
T_LEVEL = "T4"
T4_COMMENT_COUNT = 50

LOW_MAX = 3
MEDIUM_MAX = 10

REASON_MODEL = "gpt-4o-mini"
EXTRACT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 15

GROUPS = ("low", "medium", "high")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def paths() -> dict[str, Path]:
    root = project_root()
    return {
        "raw_query_jsonl": root / "data/raw/POOL-EN",
        "manifest": root / "results/tables/pool_en_profile_manifest.csv",
        "t4_truncated": root / "data/esrc/pool_en/truncated_queries/T4",
        "group_profiles_root": root / "data/diversity_groups",
        "group_summaries_root": root / "data/diversity_groups/summaries",
        "group_embeddings_root": root / "data/diversity_groups/embeddings",
        "main_t4_summaries": root / "data/esrc/pool_en/summaries/T4",
        "candidate_summaries": root / "data/esrc/pool_en/candidate_summaries",
        "candidate_embeddings": root / "data/esrc/pool_en/embeddings/candidate_embeddings.npy",
        "candidate_index": root / "results/tables/pool_en_candidate_embeddings_index.csv",
        "group_manifest": root / "results/tables/diversity_group_manifest.csv",
        "faiss_out": root / "results/tables/diversity_faiss_top15.csv",
        "reason_out": root / "results/tables/diversity_reason_predictions.csv",
        "curve_out": root / "results/tables/diversity_precision_recall_curve.csv",
        "recall_out": root / "results/tables/diversity_recall_at_precision.csv",
        "summary_out": root / "results/tables/diversity_summary.csv",
        "extract_log": root / "results/tables/diversity_extract_log.csv",
        "figures_dir": root / "results/figures",
        "summarize_prompt": root / "prompts/summarization_lermen_g2.txt",
        "reason_prompt": root / "prompts/record_selection_lermen_g2.txt",
    }
