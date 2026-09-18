"""Evaluate saved fixed-time BPM snapshots with an explicit QA exception.

Does not edit mechanical assessments, frozen selections, or model checkpoints.
The sole permitted failure is named below; all other quality gates still apply.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
CODES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODES/'ml'), str(CODES/'postprocessing'), str(CODES/'independent')]
from dataset import graph_features, save_dataset
from campaign import digest, evaluate, write_csv, write_json
from posttovtk import frames
from train import scores

FLAGGED = 'independent_interpolation_dp2_N0075_Df2.4_kf0.95_rep05'


def assess(assessment):
    if assessment['passed']:
        return
    if assessment['case'] != FLAGGED:
        raise ValueError('Unexpected failed case')
    a = assessment
    if not (a['bonds_unchanged'] and a['particle_snapshots']==3 and a['bond_snapshots']==3
            and a['initial_bonds']==a['final_bonds']
            and a['max_overlap_over_diameter'] < 1e-3
            and a['aligned_rms_displacement_over_diameter'] < 1e-2
            and np.isclose(a['ke_ratio'], 0.0013550919756306525, rtol=1e-10)):
        raise ValueError('Flagged case has an unexpected mechanical failure')


def main(root):
    root = root.resolve()
    if (root/'evaluation_complete.json').exists():
        raise ValueError('Refusing to overwrite a completed test evaluation')
    frozen = json.loads((root/'freeze.json').read_text())
    records = []
    # Record the amendment before computing any prediction errors.
    write_json(root/'fixed_time_amendment.json', dict(
        source_run=35326183899, flagged_case=FLAGGED,
        decision='User requested evaluation after the kinetic-energy gate failure was disclosed.',
        primary='All 180 finite snapshots at the registered 16 us time; no exclusions.',
        sensitivity='Also report interpolation scores excluding the flagged case.',
        unchanged='Models, selections, geometries, DEM settings, contact tolerance and exposure rays.',
        mechanical_status='179 passed, one failed the kinetic-energy criterion; not equilibrium-certified.',
        original_freeze_sha256=digest(root/'freeze.json')))
    for c in frozen['planned_cases']:
        folder = root/'bonded'/c['case']
        assessment = json.loads((folder/'assessment.json').read_text())
        assess(assessment)
        config = json.loads((folder/'case.json').read_text())
        info = max(json.loads((folder/'post/exposure/summary.json').read_text()),key=lambda s:s['step'])
        if info['step']!=160000 or info['rays']!=2048 or config['parameters_SI']['dt_s']!=1e-10:
            raise ValueError('Fixed time or exposure settings changed')
        dump = folder/f"post/post_{info['step']}.txt"
        if digest(dump)!=info['source_dump_sha256']:
            raise ValueError('Stale exposure')
        snapshots = list(frames(dump))
        if len(snapshots)!=1:
            raise ValueError('Multiple snapshots')
        step, time, columns, rows, _ = snapshots[0]
        if step!=160000 or not np.isclose(time,16.0):
            raise ValueError('Unexpected snapshot time')
        arr=np.array(rows,float); arr=arr[np.argsort(arr[:,columns.index('id')])]
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite DEM state')
        ids=arr[:,columns.index('id')].astype(int)
        if len(ids)!=c['n'] or len(set(ids))!=len(ids):
            raise ValueError('Particle identities/count changed')
        with (folder/f"post/exposure/exposure_{step}.csv").open() as f:
            exposure_rows=list(csv.DictReader(f))
        exposure={int(r['id']):float(r['geometric_exposure']) for r in exposure_rows}
        if len(exposure_rows)!=len(exposure) or set(exposure)!=set(ids):
            raise ValueError('Exposure identity mismatch')
        y=np.array([exposure[i] for i in ids])
        if not np.isfinite(y).all() or np.any((y<0)|(y>1)) or not np.isclose(y.mean(),info['mean_geometric_exposure'],atol=1e-12,rtol=0):
            raise ValueError('Invalid target')
        xyz=arr[:,[columns.index(k) for k in ('x','y','z')]]
        radii=arr[:,columns.index('radius')]
        g=graph_features(xyz,radii,1e-6)
        meta=config['source_geometry']
        g.update(case=c['case'],method=meta['method'],n=len(ids),group=meta['parameter_group'],
            diameter_um=c['diameter_um'],requested_df=c['requested_df'],requested_kf=c['requested_kf'],
            study_subset=c['subset'],target=float(y.mean()),snapshot='final',
            dump_sha256=digest(dump),rays=2048,mechanical_passed=assessment['passed'],
            mechanical_exception='kinetic-energy gate at fixed time' if not assessment['passed'] else '')
        records.append(g)
    dataset=root/'test_dataset.json'
    save_dataset(records,dataset,1e-6)
    from types import SimpleNamespace
    evaluate(SimpleNamespace(output=root,dataset=dataset))
    with (root/'test_predictions.csv').open() as f:
        predictions=list(csv.DictReader(f))
    sensitivity=[]
    for seed in frozen['protocol']['training_seeds']:
        for model in sorted({r['model'] for r in predictions}):
            rr=[r for r in predictions if r['subset']=='interpolation' and int(r['seed'])==seed
                and r['model']==model and r['case']!=FLAGGED]
            if len(rr)!=89:
                raise ValueError('Unexpected sensitivity coverage')
            sensitivity.append(dict(subset='interpolation_without_flagged_case',seed=seed,
                model=model,n=len(rr),**scores([float(r['target']) for r in rr],[float(r['prediction']) for r in rr])))
    write_csv(root/'sensitivity_metrics.csv',sensitivity)
    report=root/'results.md'
    main_report=report.read_text()
    warning=('# Fixed-time evaluation amendment\n\n'
        'Primary scores include all 180 cases. One interpolation case failed the original '
        'kinetic-energy gate (ratio 0.00135509 versus <0.001). All other mechanical checks passed. '
        'This exception was recorded before evaluating prediction errors. Models were not retrained. '
        'Treat results as prediction at 16 us, not certified equilibrium.\n\n')
    lines=['','## Sensitivity: interpolation without the flagged case (89 cases)','',
        '| Model | R² | RMSE | MAE |','|---|---:|---:|---:|']
    for model in sorted({r['model'] for r in sensitivity}):
        rr=[r for r in sensitivity if r['model']==model]
        cells=[f"{np.mean([r[k] for r in rr]):.5f} ± {np.std([r[k] for r in rr],ddof=1):.5f}" for k in ('r2','rmse','mae')]
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    report.write_text(warning+main_report+'\n'.join(lines)+'\n')
    print(report.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',type=Path)
    main(p.parse_args().experiment)
