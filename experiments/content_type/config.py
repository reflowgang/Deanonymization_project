from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
T_LEVEL = "T4"
T4_COMMENT_COUNT = 50
PROFILE_COMMENT_COUNT = 50
MIN_COMMENTS_PER_TYPE = 50

CLASSIFY_MODEL = "gpt-4o-mini"
EXTRACT_MODEL = "gpt-4o-mini"
REASON_MODEL = "gpt-4o-mini"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 15

CONTENT_TYPES = ("P", "O", "T")
TYPE_LABELS = {
    "P": "personal",
    "O": "opinion",
    "T": "topical",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def paths() -> dict[str, Path]:
    root = project_root()
    return {
        "raw_query_jsonl": root / "data/raw/POOL-EN",
        "manifest": root / "results/tables/pool_en_profile_manifest.csv",
        "t4_truncated": root / "data/esrc/pool_en/truncated_queries/T4",
        "type_profiles_root": root / "data/content_types",
        "type_summaries_root": root / "data/content_types/summaries",
        "type_embeddings_root": root / "data/content_types/embeddings",
        "candidate_summaries": root / "data/esrc/pool_en/candidate_summaries",
        "candidate_embeddings": root / "data/esrc/pool_en/embeddings/candidate_embeddings.npy",
        "candidate_index": root / "results/tables/pool_en_candidate_embeddings_index.csv",
        "classify_prompt": root / "prompts/content_type_classification.txt",
        "summarize_prompt": root / "prompts/summarization_lermen_g2.txt",
        "reason_prompt": root / "prompts/record_selection_lermen_g2.txt",
        "classifications_csv": root / "results/tables/content_type_classifications.csv",
        "qualification_csv": root / "results/tables/content_type_qualification.csv",
        "classify_log_csv": root / "results/tables/content_type_classify_log.csv",
        "faiss_out": root / "results/tables/content_type_faiss_top15.csv",
        "reason_out": root / "results/tables/content_type_reason_predictions.csv",
        "curve_out": root / "results/tables/content_type_precision_recall_curve.csv",
        "recall_out": root / "results/tables/content_type_recall_at_precision.csv",
        "summary_out": root / "results/tables/content_type_summary.csv",
        "extract_log": root / "results/tables/content_type_extract_log.csv",
        "bootstrap_ci": root / "results/tables/content_type_bootstrap_ci.csv",
        "significance_tests": root / "results/tables/content_type_significance_tests.csv",
        "figures_dir": root / "results/figures",
    }


def type_folders() -> tuple[str, ...]:
    return tuple(TYPE_LABELS[label] for label in CONTENT_TYPES)
