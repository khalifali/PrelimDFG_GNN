"""Explain latent-size choices from inner validation, independently of test scores."""
import csv,json
from pathlib import Path

def selection_report(root):
    root=Path(root)
    with (root/'selections.csv').open() as f: source=list(csv.DictReader(f))
    rows=[]
    for kind in ('gnn','ann'):
        for fold,seed in sorted({(int(r['fold']),int(r['seed'])) for r in source}):
            part={int(r['model'].split('_q')[1]):json.loads(r['selection']) for r in source
                  if r['fold']==str(fold) and r['seed']==str(seed) and r['model'].startswith(kind+'_q')}
            best=max(x['inner_r2'] for x in part.values())
            selected=min(q for q,x in part.items() if x['inner_r2']>=best-.02)
            for q,x in sorted(part.items()):
                rows.append(dict(model=kind,fold=fold,seed=seed,q=q,inner_r2=x['inner_r2'],
                    inner_rmse=x.get('inner_rmse',''),inner_mae=x.get('inner_mae',''),
                    best_epoch=x['best_epoch'],gap_from_best=best-x['inner_r2'],
                    within_tolerance=x['inner_r2']>=best-.02,selected=q==selected,
                    all_candidates_nonpositive=best<=0))
    with (root/'latent_selection.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Internal latent-size selection','',
      'Test scores in comparison.csv describe prediction performance; they do not choose q.',
      'For each fold and training seed, compare q=1,2,3,4 on the same inner validation cases.',
      'Select the smallest q whose validation R² is no more than 0.02 below the best.',
      'This is a relative compactness rule, not an absolute accuracy guarantee. A nonpositive',
      'best R² is flagged; it does not establish that any latent size is scientifically adequate.','',
      '| Model | Fold | Seed | q1 validation R² | q2 | q3 | q4 | Selected q |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,fold,seed in sorted({(x['model'],x['fold'],x['seed']) for x in rows}):
        p=[x for x in rows if (x['model'],x['fold'],x['seed'])==(model,fold,seed)]
        vals={x['q']:x for x in p};q=next(x['q'] for x in p if x['selected'])
        lines.append(f'| {model} | {fold} | {seed} | '+ ' | '.join(f"{vals[k]['inner_r2']:.4f}" if k in vals else '—' for k in (1,2,3,4))+f' | {q} |')
    (root/'latent_selection.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    import sys
    selection_report(sys.argv[1])
