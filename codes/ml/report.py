"""Produce comparison tables and figures only from held-out predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def report(root):
    root=Path(root)
    provenance=json.loads((root/'provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Training is incomplete; do not label partial predictions final results')
    rows=read_rows(root/'predictions.csv')
    models=sorted({r['model'] for r in rows})
    seeds=sorted({int(r['seed']) for r in rows})
    per_seed=[]
    for model in models:
        for seed in seeds:
            part=[r for r in rows if r['model']==model and int(r['seed'])==seed]
            if len({r['case'] for r in part}) != len(part):
                raise ValueError('An agglomerate has duplicate held-out predictions')
            expected=set(provenance['cases'])
            if provenance['arguments']['mode']=='size500':
                expected={r['case'] for r in rows if int(r['n'])==500}
            if {r['case'] for r in part} != expected:
                raise ValueError('Missing or inconsistent held-out cases')
            y=np.array([float(r['target']) for r in part]); p=np.array([float(r['prediction']) for r in part])
            per_seed.append(dict(model=model,seed=seed,n=len(y),rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p))))
    write_rows(root/'metrics_by_seed.csv',per_seed)
    summary=[]
    for model in models:
        part=[r for r in per_seed if r['model']==model]
        summary.append(dict(model=model,**{f'{key}_{stat}':float(getattr(np,stat)([r[key] for r in part],**({'ddof':1} if stat=='std' and len(part)>1 else {}))) for key in ('rmse','mae','r2') for stat in ('mean','std')}))
    write_rows(root/'comparison.csv',summary)
    title='SMOKE TEST — not scientific results' if provenance['smoke'] else 'Held-out model comparison'
    lines=[f'# {title}','','Each agglomerate is evaluated only when its group is withheld. Each seed has a complete set of held-out predictions. The table reports mean ± sample standard deviation across seeds; this is not a confidence interval.','','| Model | RMSE | MAE | R² |','|---|---:|---:|---:|']
    for r in summary:
        lines.append('| '+r['model']+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in ('rmse','mae','r2'))+' |')
    lines+=['','The six descriptors are particle count, radius of gyration divided by diameter, contact count, contact density, mean coordination and maximum coordination. Some are mathematically dependent; the linear baseline uses a least-squares pseudoinverse.','','A latent vector contains learned artificial structural parameters that replace the traditional descriptors at the prediction head. Its axes differ between separately trained models, so axes must not be pooled across folds.','','GNN dimension selection and stopping use inner validation only. Fixed-q rows are prespecified comparisons, not a basis for choosing a winner on the test set. Predictions are not clipped to [0,1].','','See provenance.json, splits.csv, selections.csv and fold_metrics.csv for provenance and between-fold variability. Historical paper scores are not substituted for these results.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(9,max(6,len(summary)*.3)))
    ax.barh([r['model'] for r in summary],[r['rmse_mean'] for r in summary],xerr=[r['rmse_std'] for r in summary])
    ax.set(xlabel='Held-out RMSE (exposure fraction)',title=title); fig.tight_layout(); fig.savefig(root/'model_comparison.png',dpi=180); plt.close(fig)
    selected=['linear','random_forest','ann','gnn_selected']
    fig,axes=plt.subplots(2,2,figsize=(8,8))
    for ax,model in zip(axes.flat,selected):
        part=[r for r in rows if r['model']==model and int(r['seed'])==seeds[0]]
        ax.scatter([float(r['target']) for r in part],[float(r['prediction']) for r in part],s=13,alpha=.6)
        ax.plot([0,1],[0,1],'k--',lw=1); ax.set(title=model,xlabel='Ray-tracing exposure',ylabel='Held-out prediction')
    fig.suptitle(f'{title}; seed {seeds[0]}'); fig.tight_layout(); fig.savefig(root/'parity.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for kind in ('gnn','ann'):
        part=sorted([r for r in summary if r['model'].startswith(kind+'_q')],key=lambda r:int(r['model'].split('_q')[1]))
        ax.errorbar([int(r['model'].split('_q')[1]) for r in part],[r['rmse_mean'] for r in part],yerr=[r['rmse_std'] for r in part],marker='o',label=kind)
    ax.set(xlabel='Number of learned parameters q',ylabel='Held-out RMSE',xticks=[1,2,3,4]); ax.legend(); fig.tight_layout(); fig.savefig(root/'latent_dimension.png',dpi=180); plt.close(fig)
    # Each curve is kept separate; refit epochs are chosen before seeing outer test cases.
    curves=root/'curves'; curves.mkdir(exist_ok=True)
    for file in sorted(root.glob('fold*/seed*/*_inner_curve.csv')):
        data=read_rows(file)
        fig,ax=plt.subplots(figsize=(6,4))
        for key in ('training_rmse','validation_rmse'):
            ax.plot([int(r['epoch']) for r in data],[float(r[key]) for r in data],label=key)
        ax.set(xlabel='Epoch',ylabel='Exposure RMSE',title='/'.join(file.parts[-3:]).replace('_inner_curve.csv','')); ax.legend(); fig.tight_layout()
        fig.savefig(curves/('_'.join(file.parts[-3:]).replace('.csv','.png')),dpi=120); plt.close(fig)
    print(f'Comparison saved to {root}/results.md',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);report(p.parse_args().results)
