"""Matched fixed-time vdW arm; initial coordinates, kicks and common parameters from BPM."""
import argparse,concurrent.futures,csv,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'postprocessing'))
from posttovtk import frames
from visualize import process

def frame(path):
    _,_,cols,rows,_=list(frames(path))[0];a=np.array(rows,float);a=a[np.argsort(a[:,cols.index('id')])]
    return cols,a

def metrics(path,initial,p):
    c,a=frame(path);x=a[:,[c.index(k) for k in ('x','y','z')]];r=a[:,c.index('radius')]
    v=a[:,[c.index(k) for k in ('vx','vy','vz')]];w=a[:,[c.index(k) for k in ('omegax','omegay','omegaz')]]
    y=x-x.mean(0);z=initial-initial.mean(0);u,_,vt=np.linalg.svd(y.T@z);q=u@vt
    if np.linalg.det(q)<0:u[:,-1]*=-1;q=u@vt
    gaps=np.linalg.norm(x[:,None]-x[None,:],axis=-1)-r[:,None]-r[None,:];np.fill_diagonal(gaps,np.inf)
    touches=gaps<=2*r[0]*1e-6;mass=p['density_kg_m3']*.001*4*np.pi*r**3/3
    return dict(rg_over_d=float(np.sqrt(np.mean(np.sum(y*y,axis=1)))/(2*r[0])),
        aligned_rms_over_d=float(np.sqrt(np.mean(np.sum((y@q-z)**2,axis=1)))/(2*r[0])),
        contacts=int(touches.sum()//2),components=int(connected_components(touches,directed=False)[0]),
        mean_coordination=float(touches.sum()/len(x)),max_coordination=int(touches.sum(1).max()),
        maximum_overlap_over_d=float(max(0,-gaps.min())/(2*r[0])),
        kinetic_fJ=float(.5*np.sum(mass[:,None]*v*v+.4*mass[:,None]*r[:,None]**2*w*w)),
        finite=bool(np.isfinite(a).all()),particles=len(a))

def one(source,dest,plugin,lmp):
    cfg=json.loads((source/'case.json').read_text());p=cfg['parameters_SI'];dest.mkdir();(dest/'post').mkdir()
    c,a=frame(source/'generated/post_0.txt');x=a[:,[c.index(k) for k in ('x','y','z')]];r=a[:,c.index('radius')];n=len(x);d=2*r[0]
    s=f'Matched vdW arm, MICRO\n\n{n} atoms\n1 atom types\n\n'
    for k,axis in enumerate('xyz'):s+=f'{x[:,k].min()-10*r[0]:.17g} {x[:,k].max()+10*r[0]:.17g} {axis}lo {axis}hi\n'
    s+='\nAtoms # sphere\n\n'
    for i,(xx,rr) in enumerate(zip(x,r),1):s+=f'{i} 1 {2*rr:.17g} {p["density_kg_m3"]*.001:.17g} '+ ' '.join(f'{v:.17g}' for v in xx)+'\n'
    (dest/'particles.data').write_text(s);shutil.copy2(source/'kick.in',dest/'kick.in')
    steps=p['steps'];dt=p['dt_s']*1e6
    script=f'''units micro
atom_style sphere
atom_modify map array
boundary f f f
newton on off
read_data particles.data
comm_modify vel yes cutoff {3*d:.17g}
neighbor {0.2*d:.17g} bin
neigh_modify every 1 delay 0 check yes
plugin load {plugin}
pair_style hybrid/overlay granular {1.2*d:.17g} lamfoam/vdw {r.max():.17g}
pair_coeff * * granular hertz/material {p['young_Pa']*.001:.17g} {p['restitution']} {p['poisson']} tangential mindlin NULL 1.0 {p['friction']} damping tsuji limit_damping
pair_coeff * * lamfoam/vdw 3.028e-5 0.0004
fix integrate all nve/sphere
fix damp_translation all viscous {p['numerical_drag_kg_s']*1e9:.17g}
fix damp_rotation all viscous/sphere {p['numerical_rot_drag_kg_m2_s']*1e21:.17g}
timestep {dt:.17g}
compute kt all ke
compute kr all erotate/sphere
variable kinetic equal c_kt+c_kr
thermo {steps//20}
thermo_style custom step time atoms c_kt c_kr v_kinetic
thermo_modify lost error format float %.16e
run 0
include kick.in
dump particles all custom {steps//2} post/post_*.txt id type x y z vx vy vz omegax omegay omegaz radius fx fy fz
dump_modify particles sort id format float %.16e time yes
run {steps}
write_restart final.restart
'''
    (dest/'in.lammps').write_text(script)
    with (dest/'run.log').open('w') as f:subprocess.run([lmp,'-in','in.lammps'],cwd=dest,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
    ci,ai=frame(dest/'post/post_0.txt');cs,asrc=frame(source/'post/post_0.txt')
    for field in ['id','x','y','z','radius','vx','vy','vz','omegax','omegay','omegaz']:
        if not np.allclose(ai[:,ci.index(field)],asrc[:,cs.index(field)],rtol=1e-13,atol=1e-14):raise ValueError('Unmatched initial state: '+field)
    exposures=process(dest/'post',2048,'pvd');old=json.loads((source/'post/exposure/summary.json').read_text())
    results=[]
    for label,folder,es in [('vdw',dest,exposures),('bpm',source,old)]:
        for item in sorted(es,key=lambda z:z['step']):
            m=metrics(folder/'post'/f"post_{item['step']}.txt",x,p)
            results.append(dict(case=source.name,model=label,step=item['step'],time_s=item['step']*p['dt_s'],exposure=item['mean_geometric_exposure'],**m))
    final=next(z for z in results if z['model']=='vdw' and z['step']==steps)
    valid=final['finite'] and final['particles']==n and final['maximum_overlap_over_d']<.001
    # vdW releases potential energy; BPM's KE/initial-KE rule is not an equilibrium test for this arm.
    (dest/'assessment.json').write_text(json.dumps(dict(passed=valid,criteria='finite, no particle loss, overlap/d<.001; fixed-time comparison, NOT equilibrium certification',metrics=final),indent=2))
    cfg['bond_model']='none; Hertz/Mindlin plus unmodified LAMFOAM lamfoam/vdw'
    cfg['vdw_parameters_SI']=dict(hamaker_J=3.028e-20,h0_m=4e-10)
    (dest/'case.json').write_text(json.dumps(cfg,indent=2))
    (dest/'paired_geometry.json').write_text(json.dumps(results,indent=2))
    if not valid:raise ValueError(f'vdW numerical check failed: {source.name}')
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--plugin',type=Path,required=True);p.add_argument('--lammps',default='lmp');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.source=a.source.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    campaign=json.loads((a.source/'campaign.json').read_text())
    if campaign['count']!=179:raise ValueError('Require the same selected 179 cases')
    (a.output/'campaign.json').write_text(json.dumps(campaign,indent=2))
    rows=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,a.source/n,a.output/n,str(a.plugin.resolve()),a.lammps):n for n in campaign['cases']}
        for f in concurrent.futures.as_completed(futures):
            try:rows+=f.result()
            except Exception as e:failures.append(dict(case=futures[f],error=str(e)));print(failures[-1],flush=True)
            print(f'Completed {len(rows)//6}/179; failures {len(failures)}',flush=True)
    if rows:
        with (a.output/'paired_geometry.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'run_summary.json').write_text(json.dumps(dict(failures=failures,all_passed=not failures,count=len(rows)//6),indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
