from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.metrics_calibration import evaluate_scores

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P2.8 evaluate estimators')
    p.add_argument('--scores-csv', type=str, required=True)
    p.add_argument('--out-csv', type=str, default=None)
    return p.parse_args()

def _as_bool(v: str) -> bool:
    return str(v).lower() in {'1', 'true', 'yes'}

def main() -> int:
    args = parse_args()
    path = Path(args.scores_csv)
    rows = []
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'ok':
                rows.append(r)
    if not rows:
        raise SystemExit('No ok rows in scores CSV')
    correct = [_as_bool(r['correct']) for r in rows]
    c_rows = [r for r in rows if r.get('c_argmax_disagrees_with_pick') not in (None, '')]
    if not c_rows and any((r.get('score_c_argmax_number') not in (None, '') for r in rows)):
        for r in rows:
            if r.get('score_c_argmax_number') in (None, '') or r.get('selected_candidate_number') in (None, ''):
                continue
            disagrees = int(int(r['score_c_argmax_number']) != int(r['selected_candidate_number']))
            r['c_argmax_disagrees_with_pick'] = str(disagrees)
        c_rows = [r for r in rows if r.get('c_argmax_disagrees_with_pick') not in (None, '')]
    n_c = len(c_rows)
    n_disagree = sum((_as_bool(r['c_argmax_disagrees_with_pick']) for r in c_rows))
    disagree_rate = n_disagree / n_c if n_c else None
    score_c_vals = [float(r['score_c']) for r in rows if r.get('score_c') not in (None, '')]
    score_c_mean = sum(score_c_vals) / len(score_c_vals) if score_c_vals else None
    score_c_median = sorted(score_c_vals)[len(score_c_vals) // 2] if score_c_vals else None
    estimators = []
    for (key, label) in [('score_a', 'a_verbalized'), ('score_b', 'b_selected_id_exp_logprob'), ('score_c', 'c_softmax_on_selected')]:
        vals = []
        ok = True
        for r in rows:
            if r.get(key) in (None, ''):
                ok = False
                break
            vals.append(float(r[key]))
        if not ok:
            print(f'skip {label}: missing values')
            continue
        m = evaluate_scores(vals, correct)
        row_out = {'estimator': label, 'n': m.n, 'n_correct': m.n_correct, 'top1_accuracy': f'{m.top1_accuracy:.6f}', 'ece': f'{m.ece:.6f}', 'brier': f'{m.brier:.6f}', 'average_precision': f'{m.average_precision:.6f}', 'recall_at_90_precision': '' if m.recall_at_90_precision is None else f'{m.recall_at_90_precision:.6f}', 'recall_at_99_precision': '' if m.recall_at_99_precision is None else f'{m.recall_at_99_precision:.6f}', 'c_argmax_disagree_n': '', 'c_argmax_disagree_rate': '', 'score_c_mean': '', 'score_c_median': '', 'score_c_uniform_baseline': ''}
        if label.startswith('c_') and disagree_rate is not None:
            row_out['c_argmax_disagree_n'] = str(n_disagree)
            row_out['c_argmax_disagree_rate'] = f'{disagree_rate:.6f}'
            row_out['score_c_mean'] = f'{score_c_mean:.6f}' if score_c_mean is not None else ''
            row_out['score_c_median'] = f'{score_c_median:.6f}' if score_c_median is not None else ''
            row_out['score_c_uniform_baseline'] = f'{1.0 / 15.0:.6f}'
        estimators.append(row_out)
        extra = ''
        if label.startswith('c_') and disagree_rate is not None:
            extra = f' | c_argmax_disagree={n_disagree}/{n_c} ({disagree_rate:.1%}) | score_c mean={score_c_mean:.4f} median={score_c_median:.4f} (uniform=1/15={1 / 15:.4f})'
        print(f'{label}: ECE={m.ece:.4f} Brier={m.brier:.4f} AP={m.average_precision:.4f} R@90P={m.recall_at_90_precision}{extra}')
    out = Path(args.out_csv) if args.out_csv else path.with_name('metrics_by_estimator.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(estimators[0].keys()))
        w.writeheader()
        w.writerows(estimators)
    print(f'Wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
