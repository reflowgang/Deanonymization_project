from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
LEVELS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
COMMENT_COUNTS: list[int | str] = [5, 10, 25, 50, 100, 200, 500, 'Full']
REDDIT_ACCURACY = [8.8, 13.4, 16.6, 25.2, 28.2, 33.2, 39.8, 39.4]
HN_ACCURACY = [4.8, 8.6, 15.6, 22.2, 26.0, 32.4, 34.6, 38.0]

def format_x_labels(levels: list[str], comment_counts: list[int | str]) -> list[str]:
    labels: list[str] = []
    for (level, count) in zip(levels, comment_counts):
        count_label = 'Full' if isinstance(count, str) else f'{count} comments'
        labels.append(f'{level}\n{count_label}')
    return labels

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / 'results/figures'
    tables_dir = project_root / 'results/tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_png = figures_dir / 'reddit_vs_hn_accuracy.png'
    out_pdf = figures_dir / 'reddit_vs_hn_accuracy.pdf'
    out_csv = tables_dir / 'reddit_vs_hn_accuracy.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['level', 'comment_count', 'reddit_accuracy', 'hn_accuracy'])
        writer.writeheader()
        for (level, count, reddit, hn) in zip(LEVELS, COMMENT_COUNTS, REDDIT_ACCURACY, HN_ACCURACY):
            writer.writerow({'level': level, 'comment_count': count, 'reddit_accuracy': reddit, 'hn_accuracy': hn})
    x = list(range(len(LEVELS)))
    x_labels = format_x_labels(LEVELS, COMMENT_COUNTS)
    (fig, ax) = plt.subplots(figsize=(10, 6))
    ax.plot(x, REDDIT_ACCURACY, marker='o', linewidth=2, label='Reddit / POOL-EN')
    ax.plot(x, HN_ACCURACY, marker='s', linewidth=2, label='Hacker News')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Deanonymization Accuracy vs Text Volume')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(max(REDDIT_ACCURACY), max(HN_ACCURACY)) * 1.1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print('Saved outputs:')
    print(f'- {out_csv}')
    print(f'- {out_png}')
    print(f'- {out_pdf}')
if __name__ == '__main__':
    main()
