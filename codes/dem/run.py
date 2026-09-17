#!/usr/bin/env python3
"""Run prepared cases, retain failed runs, and report measured geometry changes."""
import argparse
import concurrent.futures
import csv
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'postprocessing'))
from posttovtk import frames


def load_frame(path):
    snapshots=list(frames(path))
    if len(snapshots)!=1:raise ValueError('Expected one snapshot per file')
    step,time,columns,rows,_=snapshots[0]
    a=np.asarray(rows,dtype=float)
    a=a[np.argsort(a[:,columns.index('id')])]
    return step,columns,a


def analyse(folder):
    config=json.loads((folder/'case.json').read_text())
    params=config['parameters_SI'];steps=params['steps']
    _,cols,start=load_frame(folder/'post'/'post_0.txt')
    _,endcols,end=load_frame(folder/'post'/f'post_{steps}.txt')
    if cols!=endcols or start.shape!=end.shape or not np.array_equal(start[:,0],end[:,0]):
        raise ValueError('Particle identities or fields changed')
    pos=[cols.index(k) for k in ['x','y','z']]
    vel=[cols.index(k) for k in ['vx','vy','vz']]
    omega=[cols.index(k) for k in ['omegax','omegay','omegaz']]
    radius=start[:,cols.index('radius')]
    m=params['density_kg_m3']*.001*4*np.pi*radius**3/3
    inertia=.4*m*radius**2
    def kinetic(a):return float(.5*np.sum(m[:,None]*a[:,vel]**2+inertia[:,None]*a[:,omega]**2))
    ke0=kinetic(start);ke1=kinetic(end)
    x=start[:,pos]-start[:,pos].mean(axis=0);y=end[:,pos]-end[:,pos].mean(axis=0)
    # Best-fit rigid rotation separates deformation from harmless rigid motion.
    u,_,vt=np.linalg.svd(y.T@x);q=u@vt
    if np.linalg.det(q)<0:u[:,-1]*=-1;q=u@vt
    diff=y@q-x
    rms=float(np.sqrt(np.mean(np.sum(diff**2,axis=1)))/(2*radius[0]))
    rg0=float(np.sqrt(np.mean(np.sum(x*x,axis=1))+.6*radius[0]**2))
    rg1=float(np.sqrt(np.mean(np.sum(y*y,axis=1))+.6*radius[0]**2))
    distance=np.linalg.norm(y[:,None]-y[None,:],axis=-1)
    ii,jj=np.triu_indices(len(y),1)
    maxover=float(max(0,np.max(radius[ii]+radius[jj]-distance[ii,jj]))/(2*radius[0]))
    bp=folder/'bond'/f'bond_{steps}.txt'
    lines=bp.read_text().splitlines();bond_count=int(lines[lines.index('ITEM: NUMBER OF ENTRIES')+1])
    initial_bonds=int(config['source_geometry']['contacts'])
    def bond_pairs(path):
        lines=path.read_text().splitlines()
        h=next(i for i,line in enumerate(lines) if line.startswith('ITEM: ENTRIES'))
        c=lines[h].split()[2:]
        pairs=[tuple(sorted((int(float(row.split()[c.index('c_bondIDs[1]')])),
                             int(float(row.split()[c.index('c_bondIDs[2]')]))))) for row in lines[h+1:] if row.strip()]
        return sorted(pairs)
    bonds_unchanged=bond_pairs(folder/'bond'/'bond_0.txt')==bond_pairs(bp)
    snapshot_count=len(list((folder/'post').glob('post_*.txt')))
    bond_snapshot_count=len(list((folder/'bond').glob('bond_*.txt')))
    finite=bool(np.isfinite(end).all())
    ratio=ke1/ke0 if ke0>0 else 0.
    result=dict(case=folder.name,n=len(y),steps=steps,final_time_s=steps*params['dt_s'],
                initial_bonds=initial_bonds,final_bonds=bond_count,bonds_unchanged=bonds_unchanged,
                particle_snapshots=snapshot_count,bond_snapshots=bond_snapshot_count,
                initial_ke_fJ=ke0,final_ke_fJ=ke1,ke_ratio=ratio,
                aligned_rms_displacement_over_diameter=rms,relative_rg_change=rg1/rg0-1,
                max_overlap_over_diameter=maxover,
                passed=bool(finite and bonds_unchanged and snapshot_count==3 and bond_snapshot_count==3 and bond_count==initial_bonds and maxover<1e-3 and rms<1e-2 and ratio<1e-3))
    (folder/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    # Post-DEM CSV is SI; raw post dumps remain MICRO for existing tooling.
    np.savetxt(folder/'final_particles.csv',np.column_stack([end[:,pos],radius])*1e-6,
               delimiter=',',header='x_m,y_m,z_m,radius_m',comments='',fmt='%.17g')
    return result


def run_case(folder, executable, compat):
    if (folder/'log.lammps').exists():
        raise FileExistsError(f'{folder}: existing run; use a fresh prepared directory')
    script=(folder/'in.template').read_text().replace('__BPM_COMPAT__',compat)
    (folder/'in.lammps').write_text(script)
    with (folder/'run.stdout').open('w') as log:
        subprocess.run([executable,'-in','in.lammps'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    return analyse(folder)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path)
    p.add_argument('--lammps',default='lmp');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    info=subprocess.check_output([args.lammps,'-h'],text=True,stderr=subprocess.STDOUT)
    match=re.search(r'Simulator - (\d+ \w+ \d{4})',info)
    if not match:raise RuntimeError('Could not identify the LAMMPS version')
    date=datetime.datetime.strptime(match.group(1),'%d %b %Y')
    compat='frame particle damping dem' if date>=datetime.datetime(2026,7,4) else ''
    if not all(style in info for style in ['bpm/sphere','bpm/rotational','granular','viscous/sphere']):
        raise RuntimeError('LAMMPS needs BPM and GRANULAR packages')
    root=args.campaign.resolve();config=json.loads((root/'campaign.json').read_text())
    (root/'lammps_help.txt').write_text(info)
    results=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,root/name,args.lammps,compat):name for name in config['cases']}
        for future in concurrent.futures.as_completed(futures):
            name=futures[future]
            try:
                row=future.result();results.append(row)
                if not row['passed']:failures.append(dict(case=name,error='Assessment thresholds not met'))
            except Exception as e:failures.append(dict(case=name,error=str(e)))
            print(f'{len(results)}/{config["count"]} completed; {len(failures)} failures',flush=True)
    results.sort(key=lambda r:r['case'])
    if results:
        with (root/'dem_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (root/'run_summary.json').write_text(json.dumps(dict(expected=config['count'],completed=len(results),
           failures=failures,lammps_version=match.group(1),bpm_compatibility=compat or 'legacy defaults',
           all_passed=len(results)==config['count'] and not failures),indent=2)+'\n')
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
