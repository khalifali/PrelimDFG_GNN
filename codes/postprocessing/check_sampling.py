"""Compare direction resolution and rigid rotation on fixed representative cases."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
from exposure import geometric_exposure
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation'))
from generate import rotation

def check(root):
    with (root/'manifest.csv').open() as f:cases=list(csv.DictReader(f))
    rows=[]
    for case in cases:
        if float(case['diameter_um'])!=1 or int(case['replicate'])!=1:continue
        data=np.loadtxt(root/case['case']/'particles.csv',delimiter=',',skiprows=1)
        xyz=data[:,:3];r=data[:,3]
        reference=geometric_exposure(xyz,r,8192)
        for sampling,rays,turn in [('fibonacci',512,0),('fibonacci',2048,0),('fibonacci',4096,0),('fibonacci',8192,0),('fibonacci',2048,1),('fibonacci',2048,2)]:
            points=xyz if turn==0 else xyz@rotation(np.random.default_rng(100+turn)).T
            values=reference if rays==8192 and turn==0 else geometric_exposure(points,r,rays)
            rows.append(dict(case=case['case'],n=case['n'],df=case['requested_df'],sampling=sampling,rays=rays,rotation=turn,
                             mean=float(values.mean()),reference_mean=float(reference.mean()),absolute_mean_difference=float(abs(values.mean()-reference.mean())),
                             maximum_particle_difference=float(abs(values-reference).max())))
        print(case['case'],'checked',flush=True)
    with (root/'sampling_checks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);check(p.parse_args().root)
