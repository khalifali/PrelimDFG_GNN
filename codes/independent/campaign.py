"""Independent BPM holdouts: freeze, fit, generate, and evaluate in separate stages.

No test dataset is read by freeze or fit. Only development INNER validation
statistics determine dimensions and training durations. Test results never
select a seed, a dimension, a checkpoint, or a contact threshold.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
sys.path[:0] = [str(CODES/'ml'), str(CODES/'generation'), str(CODES/'postprocessing')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty output')
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def planned_cases(protocol):
    for subset, settings in protocol['sets'].items():
        for d in protocol['diameters_um']:
            for n in settings['counts']:
                for df, kf in settings['morphologies']:
                    for rep in range(1, protocol['replicates']+1):
                        case = f'independent_{subset}_dp{d:g}_N{n:04d}_Df{df:g}_kf{kf:g}_rep{rep:02d}'
                        key = f"{protocol['master_seed']}:{protocol['seed_namespace']}:{case}"
                        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                        yield dict(case=case, subset=subset, n=n, diameter_um=d,
                                   requested_df=df, requested_kf=kf, replicate=rep, seed=seed)


def freeze(a):
    p = json.loads((HERE/'protocol.json').read_text())
    dev = json.loads(a.dataset.read_text())
    records = dev['records']
    if len(records) != p['development_cases'] or any(g['snapshot'] != 'final' or g['rays'] != p['rays'] for g in records):
        raise ValueError('Unexpected development dataset')
    if dev['contact_tolerance_over_d'] != p['contact_tolerance_over_d']:
        raise ValueError('Contact tolerance differs')
    groups = defaultdict(list)
    with a.selections.open() as f:
        for r in csv.DictReader(f):
            groups[r['model']].append(json.loads(r['selection']))
    durations = {}; selected = {}; validation = {}
    for kind in ('gnn', 'ann'):
        for q in p['latent_dimensions']:
            name = f'{kind}_q{q}'
            if len(groups[name]) != 15:
                raise ValueError('Expected five development folds and three seeds')
            validation[name] = statistics.mean(r['inner_r2'] for r in groups[name])
            durations[name] = max(1, round(statistics.median(r['best_epoch'] for r in groups[name])))
        best = max(validation[f'{kind}_q{q}'] for q in p['latent_dimensions'])
        selected[kind] = min(q for q in p['latent_dimensions'] if validation[f'{kind}_q{q}'] >= best-.02)
    durations['ann'] = max(1, round(statistics.median(r['best_epoch'] for r in groups['ann'])))
    rf = Counter((r['min_samples_leaf'], r['max_features']) for r in groups['random_forest'])
    leaf, features = sorted(rf, key=lambda x: (-rf[x], x))[0]
    cases = list(planned_cases(p))
    if {g['case'] for g in records} & {c['case'] for c in cases}:
        raise ValueError('Case identity overlap')
    a.output.mkdir(parents=True, exist_ok=False)
    code_files = [HERE/'campaign.py', HERE/'protocol.json', CODES/'ml/train.py',
                  CODES/'ml/models.py', CODES/'ml/dataset.py', CODES/'generation/generate.py',
                  CODES/'generation/tunable.py', CODES/'dem/prepare.py', CODES/'dem/run.py',
                  CODES/'postprocessing/exposure.py', CODES/'postprocessing/visualize.py']
    write_json(a.output/'freeze.json', dict(protocol=p, development_sha256=digest(a.dataset),
        selection_sha256=digest(a.selections), selected_q=selected, epochs=durations,
        inner_mean_r2=validation, forest=dict(min_samples_leaf=leaf, max_features=features),
        source_sha256={str(f.relative_to(CODES)):digest(f) for f in code_files},
        planned_cases=cases, selection_data='development inner validation only'))
    write_csv(a.output/'planned_cases.csv', cases)
    print(json.dumps(dict(selected_q=selected, epochs=durations, test_counts=dict(Counter(c['subset'] for c in cases)))), flush=True)


def fit(a):
    import numpy as np
    import torch
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from dataset import read_dataset
    from train import train_neural, save_checkpoint
    frozen = json.loads((a.output/'freeze.json').read_text())
    if digest(a.dataset) != frozen['development_sha256']:
        raise ValueError('Development data changed')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    records = read_dataset(a.dataset)
    x = np.stack([g['descriptors'] for g in records]); y = np.array([g['target'] for g in records])
    root = a.output/'models'; root.mkdir(exist_ok=False)
    for seed in frozen['protocol']['training_seeds']:
        folder = root/f'seed{seed}'; folder.mkdir()
        for name, epochs in frozen['epochs'].items():
            kind = name.split('_')[0]; q = int(name.split('_q')[1]) if '_q' in name else 0
            model, scale, _, history = train_neural(records, None, kind, q, seed+1000*q, epochs, 50, fixed_epochs=epochs)
            save_checkpoint(folder/f'{name}.pt', model, scale, kind, q, epochs, records)
            write_csv(folder/f'{name}_curve.csv', history)
            print(f'Full development fit: seed={seed} {name}, epochs={epochs}', flush=True)
        baselines = dict(linear=make_pipeline(StandardScaler(), LinearRegression()),
            random_forest=RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=2, **frozen['forest']),
            random_forest_v9=RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=seed, n_jobs=2))
        for name, model in baselines.items():
            model.fit(x, y); joblib.dump(model, folder/f'{name}.joblib')
    write_json(a.output/'models_frozen.json', dict(freeze_sha256=digest(a.output/'freeze.json'),
        files={str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.suffix in ('.pt','.joblib')}))


def signature(points):
    """Rotation/translation/particle-order/scale invariant duplicate fingerprint."""
    import numpy as np
    from scipy.spatial.distance import pdist
    distances = np.sort(pdist(points))
    # Input coordinates are in primary-radius units; quantization ignores roundoff.
    return hashlib.sha256(np.round(distances, 7).astype('<f8').tobytes()).hexdigest()


def generate(a):
    import numpy as np
    from generate import inspect, write_vtk
    from tunable import generate as generate_points
    frozen = json.loads((a.output/'freeze.json').read_text())
    locked = json.loads((a.output/'models_frozen.json').read_text())
    if locked['freeze_sha256'] != digest(a.output/'freeze.json'):
        raise ValueError('Protocol changed after fitting')
    from posttovtk import frames
    seen = set(); source_seeds = set()
    for file in sorted(a.source.glob('*/generated/post_0.txt')):
        _, _, cols, rows, _ = list(frames(file))[0]
        arr = np.array(rows, float)
        xyz = arr[:, [cols.index(k) for k in ('x','y','z')]]
        seen.add(signature(xyz/arr[0,cols.index('radius')]))
        source_seeds.add(int(json.loads((file.parents[1]/'case.json').read_text())['source_geometry']['seed']))
    if len(source_seeds) != 179:
        raise ValueError('Expected all 179 original geometry seeds for duplicate audit')
    root = a.output/'geometries'; root.mkdir(exist_ok=False)
    records = []
    for c in frozen['planned_cases']:
        if c['seed'] in source_seeds:
            raise ValueError('Generation seed collision')
        points = generate_points(c['n'], c['requested_df'], c['requested_kf'], np.random.default_rng(c['seed']))
        sig = signature(points)
        if sig in seen:
            raise ValueError('Duplicate or scaled copy of development/another test geometry')
        seen.add(sig)
        metrics, edges = inspect(points)
        target = (c['n']/c['requested_kf'])**(1/c['requested_df'])
        error = abs(metrics['rg_over_radius']/target-1)
        if error > 1e-9:
            raise ValueError('Mass-radius constraint failed')
        folder = root/c['case']; folder.mkdir()
        radius = c['diameter_um']*.5e-6; xyz = points*radius
        np.savetxt(folder/'particles.csv', np.column_stack([xyz,np.full(c['n'],radius)]), delimiter=',', header='x_m,y_m,z_m,radius_m', comments='', fmt='%.17g')
        np.savetxt(folder/'contacts.csv', edges+1, delimiter=',', header='id_i,id_j', comments='', fmt='%d')
        write_vtk(folder/'particles.vtk', xyz, radius)
        row = dict(**c, method='tunable_fractal', family='tunable_fractal', detector_radius_over_r='',
            sample=c['replicate'], study_subset=c['subset'],
            parameter_group=f"tunable_fractal|N{c['n']}|Df{c['requested_df']}|kf{c['requested_kf']}",
            mass_radius_relative_error=error, geometry_fingerprint=sig)
        row.update(metrics)
        write_json(folder/'metadata.json', row); records.append(row)
        write_csv(root/'manifest.csv', records)
        print(f"Generated {len(records)}/{len(frozen['planned_cases'])}: {c['case']}", flush=True)
    write_json(root/'audit.json', dict(count=len(records), development_count=179,
        seed_overlap=0, geometric_duplicates=0, all_geometry_checks_passed=True,
        fingerprint='SHA256 of sorted pair distances in radius units, rounded to 7 decimals'))


def evaluate(a):
    import numpy as np
    import torch
    import joblib
    from dataset import read_dataset
    from predict import load_checkpoint
    from models import predict
    from train import scores
    torch.set_num_threads(2)
    frozen = json.loads((a.output/'freeze.json').read_text())
    locked = json.loads((a.output/'models_frozen.json').read_text())
    if locked['freeze_sha256'] != digest(a.output/'freeze.json'):
        raise ValueError('Frozen protocol changed')
    root = a.output/'models'
    for path, sha in locked['files'].items():
        if digest(root/path) != sha:
            raise ValueError('Model changed since pre-test freeze')
    records = read_dataset(a.dataset)
    planned = {c['case']:c for c in frozen['planned_cases']}
    if len(records) != len(planned) or {g['case'] for g in records} != set(planned):
        raise ValueError('Missing/extra test cases; no silent exclusions')
    for g in records:
        c = planned[g['case']]
        if g['n'] != c['n'] or g['study_subset'] != c['subset'] or g['rays'] != frozen['protocol']['rays']:
            raise ValueError('Test metadata differs from registered design')
    rows = []; metrics = []
    for subset in frozen['protocol']['sets']:
        test = [g for g in records if g['study_subset']==subset]
        y = np.array([g['target'] for g in test]); x = np.stack([g['descriptors'] for g in test])
        for seed in frozen['protocol']['training_seeds']:
            predictions = {}
            for name in frozen['epochs']:
                model, scale, checkpoint = load_checkpoint(root/f'seed{seed}'/f'{name}.pt')
                if set(checkpoint['training_cases']) & set(planned):
                    raise ValueError('Test cases present in model training')
                predictions[name] = predict(model, test, scale, checkpoint['kind'])[0]
            for kind, q in frozen['selected_q'].items():
                predictions[f'{kind}_selected'] = predictions[f'{kind}_q{q}']
            for name in ('linear','random_forest','random_forest_v9'):
                predictions[name] = joblib.load(root/f'seed{seed}'/f'{name}.joblib').predict(x)
            for name, values in predictions.items():
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite prediction')
                metrics.append(dict(subset=subset, seed=seed, model=name, n=len(test), **scores(y,values)))
                for g, value in zip(test,values):
                    rows.append(dict(subset=subset, seed=seed, model=name, case=g['case'], n=g['n'],
                        group=g['group'], target=g['target'], prediction=float(value)))
    write_csv(a.output/'test_predictions.csv',rows)
    write_csv(a.output/'test_metrics.csv',metrics)
    by_setting = []
    for subset in frozen['protocol']['sets']:
        for seed in frozen['protocol']['training_seeds']:
            for name in sorted({r['model'] for r in rows}):
                chosen = [r for r in rows if r['subset']==subset and r['seed']==seed and r['model']==name]
                for group in sorted({r['group'] for r in chosen}):
                    rr = [r for r in chosen if r['group']==group]
                    by_setting.append(dict(subset=subset, seed=seed, model=name, group=group,
                        n=len(rr), **scores([r['target'] for r in rr],[r['prediction'] for r in rr])))
    write_csv(a.output/'test_metrics_by_parameter.csv',by_setting)
    lines = ['# Independent BPM evaluation', '',
        'Models were fitted on the 179 development cases before test generation. '
        'Dimension and epoch selection used development inner-validation statistics only. '
        'All fixed-q models are prespecified comparisons, not candidates for test-based selection.', '',
        'Mean ± sample SD across three training seeds is training variability, not a confidence interval. '
        'Parameter-specific metrics are in test_metrics_by_parameter.csv. '
        'Interpolation and size extrapolation use the same reconstructed generator and fixed 16 us mechanical check; '
        'neither establishes equilibrium or experimental generalization.', '',
        f"Selected latent dimensions: {frozen['selected_q']}", '']
    for subset in frozen['protocol']['sets']:
        lines += [f'## {subset}', '', '| Model | R² | RMSE | MAE |', '|---|---:|---:|---:|']
        for name in sorted({r['model'] for r in metrics}):
            rr = [r for r in metrics if r['subset']==subset and r['model']==name]
            cells = [f"{np.mean([r[k] for r in rr]):.5f} ± {np.std([r[k] for r in rr],ddof=1):.5f}" for k in ('r2','rmse','mae')]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines.append('')
    (a.output/'results.md').write_text('\n'.join(lines))
    write_json(a.output/'evaluation_complete.json', dict(test_sha256=digest(a.dataset),
        freeze_sha256=digest(a.output/'freeze.json'), count=len(records), status='complete'))
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','fit','generate','evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--selections', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    globals()[args.stage](args)
