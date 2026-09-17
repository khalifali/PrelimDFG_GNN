"""Build paired, provenance-checked graphs from completed DEM/exposure outputs."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'postprocessing'))
from posttovtk import frames

DESCRIPTORS = ['N', 'Rg_over_d', 'Nc', 'contact_density', 'mean_coordination', 'max_coordination']


def graph_features(xyz, radii, tolerance=1e-6):
    xyz, radii = np.asarray(xyz, float), np.asarray(radii, float)
    n = len(radii)
    if n < 2 or xyz.shape != (n, 3) or not np.isfinite(xyz).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Invalid particle geometry')
    if not np.allclose(radii, radii[0], rtol=1e-6, atol=0):
        raise ValueError('This study uses monodisperse particles only')
    d = 2 * np.median(radii)
    pos = (xyz - xyz.mean(axis=0)) / d
    delta = pos[:, None, :] - pos[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    contacts = distance <= (radii[:, None] + radii[None, :]) / d + tolerance
    np.fill_diagonal(contacts, False)
    src, dst = np.nonzero(contacts)
    degree = contacts.sum(axis=1)
    rg = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
    nc = len(src) // 2
    descriptors = np.array([n, rg, nc, 2*nc/(n*(n-1)), degree.mean(), degree.max()], np.float32)
    if rg <= 0:
        raise ValueError('Coincident particle centers')
    nodes = np.column_stack([np.linalg.norm(pos, axis=1)/rg, degree]).astype(np.float32)
    return dict(pos=pos.astype(np.float32), nodes=nodes, edge=np.array([src, dst], dtype=np.int64), descriptors=descriptors)


def load_campaign(root, snapshot='final', tolerance=1e-6):
    root = Path(root)
    records = []
    for file in sorted(root.glob('*/case.json')):
        case = json.loads(file.read_text())
        meta = case['source_geometry']
        assessment = json.loads((file.parent / 'assessment.json').read_text())
        # run.py has already assessed the complete campaign. Do not infer pass from presence alone.
        if not assessment.get('passed', False):
            raise ValueError(f'{file.parent.name}: DEM assessment failed')
        summaries = json.loads((file.parent/'post/exposure/summary.json').read_text())
        info = sorted(summaries, key=lambda s: s['step'])[0 if snapshot == 'initial' else -1]
        if snapshot == 'final' and info['step'] != case['parameters_SI']['steps']:
            raise ValueError('Final exposure does not match configured final DEM step')
        dump = file.parent / f"post/post_{info['step']}.txt"
        digest = hashlib.sha256(dump.read_bytes()).hexdigest()
        if digest != info['source_dump_sha256']:
            raise ValueError(f'Stale exposure: {dump}')
        snapshots = list(frames(dump))
        if len(snapshots) != 1 or snapshots[0][0] != info['step']:
            raise ValueError('Expected one matching snapshot')
        _, _, columns, rows, _ = snapshots[0]
        arr = np.array(rows, float)
        ids = arr[:, columns.index('id')].astype(int)
        order = np.argsort(ids)
        ids = ids[order]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate particle IDs')
        xyz = arr[order][:, [columns.index(k) for k in ('x', 'y', 'z')]]
        radii = arr[order, columns.index('radius')]
        with (file.parent / f"post/exposure/exposure_{info['step']}.csv").open() as f:
            exposure_rows = list(csv.DictReader(f))
        exposure = {int(r['id']): float(r['geometric_exposure']) for r in exposure_rows}
        if len(exposure_rows) != len(exposure) or set(exposure) != set(ids):
            raise ValueError('Exposure IDs do not match geometry')
        y = np.array([exposure[i] for i in ids])
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)) or not np.isclose(y.mean(), info['mean_geometric_exposure'], atol=1e-12, rtol=0):
            raise ValueError('Invalid exposure target')
        g = graph_features(xyz, radii, tolerance)
        g.update(case=meta['case'], method=meta['method'], n=len(ids),
                 group=meta.get('parameter_group') or f"{meta['method']}|{meta['n']}|{meta['detector_radius_over_r']}",
                 diameter_um=float(meta['diameter_um']),
                 requested_df=meta.get('requested_df',''),requested_kf=meta.get('requested_kf',''),
                 study_subset=meta.get('study_subset','legacy'),
                 target=float(y.mean()), snapshot=snapshot, dump_sha256=digest, rays=info['rays'])
        records.append(g)
    if not records or len({r['case'] for r in records}) != len(records):
        raise ValueError('Missing or duplicate cases')
    return records


def save_dataset(records, path, tolerance):
    # JSON is portable and requires no unsafe pickle loading.
    data = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in g.items()} for g in records]
    Path(path).write_text(json.dumps(dict(schema=1, descriptors=DESCRIPTORS, contact_tolerance_over_d=tolerance, records=data)))


def read_dataset(path):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 1 or data['descriptors'] != DESCRIPTORS:
        raise ValueError('Unsupported dataset schema')
    for g in data['records']:
        for key in ('pos', 'nodes', 'descriptors', 'edge'):
            g[key] = np.array(g[key], dtype=np.int64 if key == 'edge' else np.float32)
    return data['records']

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot', choices=['initial','final'], default='final')
    p.add_argument('--contact-tolerance', type=float, default=1e-6)
    p.add_argument('--expected-cases', type=int, default=180)
    a = p.parse_args()
    records = load_campaign(a.campaign, a.snapshot, a.contact_tolerance)
    if len(records) != a.expected_cases:
        raise ValueError(f'Expected {a.expected_cases} cases, found {len(records)}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(records, a.output, a.contact_tolerance)
    print(f'Saved {len(records)} {a.snapshot} graphs, {len(set(g["group"] for g in records))} groups to {a.output}', flush=True)
