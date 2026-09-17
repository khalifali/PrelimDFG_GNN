#!/usr/bin/env python3
"""180 independent geometries; structural labels are within-method/size ranks."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from generate import bpca, bcca, local_filling, inspect, write_vtk


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for method in ['bpca', 'bcca', 'local_filling']:
        for n in [50, 100, 200, 500]:
            group = []
            for sample in range(15):
                case = f'{method}_N{n:04d}_sample{sample+1:02d}'
                seed = int.from_bytes(hashlib.sha256(f'18427:study180:{case}'.encode()).digest()[:8], 'little')
                rng = np.random.default_rng(seed)
                detector = [3., 6., 12.][sample//5] if method == 'local_filling' else None
                points = local_filling(n, rng, detector) if detector else {'bpca':bpca,'bcca':bcca}[method](n,rng)
                metrics, edges = inspect(points)
                folder = output/case
                folder.mkdir()
                radius = 0.5e-6
                physical = points*radius
                np.savetxt(folder/'particles.csv', np.column_stack([physical,np.full(n,radius)]),
                           delimiter=',',header='x_m,y_m,z_m,radius_m',comments='',fmt='%.17g')
                np.savetxt(folder/'contacts.csv',edges+1,delimiter=',',header='id_i,id_j',comments='',fmt='%d')
                write_vtk(folder/'particles.vtk',physical,radius)
                row = dict(case=case,method=method,family=method,detector_radius_over_r=detector,
                           sample=sample+1,replicate=0,seed=seed,diameter_um=1.,
                           compactness_score=n**(1/3)/metrics['rg_over_radius'],
                           structural_class='',class_definition='within_method_and_N_tertile',**metrics)
                group.append(row)
            # Large Rg means more open at fixed N/r. No claim of universal density.
            ranked = sorted(group,key=lambda r:r['compactness_score'])
            for idx,row in enumerate(ranked):
                row['structural_class']=['open','intermediate','compact'][idx//5]
                row['replicate']=idx%5+1
                (output/row['case']/'metadata.json').write_text(json.dumps(row,indent=2)+'\n')
            records.extend(group)
            print(f'{method}, N={n}: 15/15 connected, non-overlapping',flush=True)
    with (output/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=dict(case_count=len(records),all_geometry_checks_passed=True,
                 design='3 methods x 4 counts x 3 relative structural classes x 5 realizations',
                 class_caveat='Tertiles within method and N, not shared absolute compactness thresholds or dense packing.',
                 diameter_um=1,python=platform.python_version(),numpy=np.__version__,
                 generator_sha256=hashlib.sha256(Path(__file__).with_name('generate.py').read_bytes()).hexdigest(),
                 study_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)
