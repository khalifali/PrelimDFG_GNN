"""Controlled original-case BPM/vdW diagnostic, not a replacement ML campaign."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'postprocessing'))
from posttovtk import frames
from visualize import process

def read_original(path):
    s=path.read_text();a=np.loadtxt(s.split('Atoms')[1].split('Velocities')[0].strip().splitlines())
    a=a[np.argsort(a[:,0])]
    v=np.loadtxt(s.split('Velocities')[1].strip().splitlines())
    if not np.array_equal(a[:,0],np.arange(1,len(a)+1)) or np.any(v[:,1:]!=0):
        raise ValueError('Expected consecutive IDs and zero initial velocities')
    return a[:,4:7],a[:,2]/2,a[:,3]

def run_case(out,xyz,r,rho,model,plugin,executable,dt=5e-5):
    # MICRO units throughout; physical duration 15 us, matching old 300000*5e-11 s.
    n=len(xyz);steps=round(15/dt);folder=out/model;folder.mkdir()
    (folder/'post').mkdir()
    dist=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
    tol=.005 if model=='bpm_near' else 1e-6
    edges=np.array([(i+1,j+1) for i in range(n) for j in range(i+1,n) if dist[i,j]<=r[i]+r[j]+tol])
    bpm=model.startswith('bpm');nb=len(edges) if bpm else 0
    text=f'Original case001; identical geometry in every branch\n\n{n} atoms\n'
    if bpm:text+=f'{nb} bonds\n'
    text+='\n1 atom types\n'+('1 bond types\n' if bpm else '')+'\n'
    for ax in 'xyz':text+=f'-100 100 {ax}lo {ax}hi\n'
    text+='\nAtoms\n\n'
    for i,(p,rr,density) in enumerate(zip(xyz,r,rho),1):
        text+=f'{i} '+('1 1 ' if bpm else '1 ')+f'{2*rr:.17g} {density:.17g} '+' '.join(f'{x:.17g}' for x in p)+'\n'
    if bpm:
        text+='\nBonds\n\n'+''.join(f'{k} 1 {i} {j}\n' for k,(i,j) in enumerate(edges,1))
    (folder/'particles.data').write_text(text)
    script=f'''units micro
atom_style {'bpm/sphere' if bpm else 'sphere'}
atom_modify map array
boundary p p p
newton on off
read_data particles.data {'extra/special/per/atom 50' if bpm else ''}
comm_modify vel yes cutoff 3.0
neighbor 0.2 bin
neigh_modify every 1 delay 0 check yes
'''
    hertz='hertz/material 1e6 0.2 0.45 tangential mindlin NULL 1.0 0.3 damping tsuji limit_damping'
    if bpm:
        script+=f'''special_bonds lj 0 1 1 coul 1 1 1
pair_style granular 1.2
pair_coeff * * {hertz}
bond_style bpm/rotational break no smooth no
bond_coeff 1 10000 4000 100 100 1 1 1 1 10 10 0.1 0.1
fix integrate all nve/bpm/sphere
'''
    else:
        script+=f'''plugin load {plugin}
pair_style hybrid/overlay granular 1.2 lamfoam/vdw 0.5
pair_coeff * * granular {hertz}
pair_coeff * * lamfoam/vdw 3.028e-5 0.0004
fix integrate all nve/sphere
'''
    # No artificial fluid drag/kick: original input uses zero velocity and contact dissipation.
    script+=f'''timestep {dt:.17g}
compute kt all ke
compute kr all erotate/sphere
variable kinetic equal c_kt+c_kr
thermo {steps//20}
thermo_style custom step time atoms c_kt c_kr v_kinetic
thermo_modify lost error format float %.16e
run 0
dump particles all custom {steps//2} post/post_*.txt id type x y z vx vy vz omegax omegay omegaz radius fx fy fz
dump_modify particles sort id format float %.16e time yes
run {steps}
write_restart final.restart
'''
    (folder/'in.lammps').write_text(script)
    with (folder/'run.log').open('w') as f:
        subprocess.run([executable,'-in','in.lammps'],cwd=folder,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
    exposure=process(folder/'post',2048,'pvd')
    records=[]
    for info in sorted(exposure,key=lambda x:x['step']):
        snap=list(frames(folder/'post'/f"post_{info['step']}.txt"))[0]
        cols=snap[2];a=np.array(snap[3],float);a=a[np.argsort(a[:,cols.index('id')])]
        x=a[:,[cols.index(k) for k in ('x','y','z')]];radii=a[:,cols.index('radius')]
        if len(x)!=n or not np.isfinite(a).all() or not np.allclose(radii,r,rtol=0,atol=1e-14):raise ValueError('Invalid final geometry')
        if info['step']==0 and not np.allclose(x,xyz,rtol=0,atol=1e-13):raise ValueError('Initial coordinates changed')
        distances=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        gaps=distances-r[:,None]-r[None,:];np.fill_diagonal(gaps,np.inf)
        touch=gaps<=1e-6;near=gaps<=.005
        xx=xyz-xyz.mean(0);yy=x-x.mean(0);u,_,vt=np.linalg.svd(yy.T@xx)
        rot=u@vt
        if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
        mass=rho*4*np.pi*r**3/3;I=.4*mass*r**2
        v=a[:,[cols.index(k) for k in ('vx','vy','vz')]];w=a[:,[cols.index(k) for k in ('omegax','omegay','omegaz')]]
        row=dict(model=model,step=info['step'],time_us=info['step']*dt,permanent_bonds=nb,
            contacts=int(touch.sum()//2),contact_components=int(connected_components(touch,directed=False)[0]),
            near_pairs=int(near.sum()//2),near_components=int(connected_components(near,directed=False)[0]),
            mean_coordination=float(touch.sum()/n),rg_over_d=float(np.sqrt(np.mean(np.sum(yy**2,axis=1)))/(2*r[0])),
            aligned_rms_over_d=float(np.sqrt(np.mean(np.sum((yy@rot-xx)**2,axis=1)))/(2*r[0])),
            maximum_overlap_over_d=float(max(0,-gaps.min())/(2*r[0])),
            mean_exposure=info['mean_geometric_exposure'],kinetic_fJ=float(.5*np.sum(mass[:,None]*v*v+I[:,None]*w*w)))
        records.append(row)
    (folder/'metrics.json').write_text(json.dumps(records,indent=2))
    print(model,records[-1],flush=True)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--plugin',type=Path,required=True);p.add_argument('--lammps',default='lmp');a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).parent/'input';xyz,r,rho=read_original(src/'original_case001.sys')
    old=np.loadtxt(src/'reported_relaxed_case001.csv',delimiter=',')
    note=dict(initial_vs_reported_relaxed_max_coordinate_difference=float(abs(xyz-old[:,:3]).max()),
        original_sha256=hashlib.sha256((src/'original_case001.sys').read_bytes()).hexdigest(),
        interpretation='Uploaded CSV and SYS coordinates are identical; no historical displacement is established.',
        bond_thresholds_over_d=dict(bpm_strict=1e-6,bpm_near=.005),
        lamfoam_commit='f61d4f06d0a5f57e7e8f962420cb86a42bdf71ba',
        model_limitations='vdW normal law from LAMFOAM; LAMMPS Mindlin contact is not an exact LIGGGHTS history/rolling/torsion reproduction; no rolling model in either arm.',
        common_parameters_SI=dict(density=1050,young=1e9,poisson=.45,restitution=.2,friction=.3,initial_velocity=0,artificial_drag=0,duration_s=15e-6),
        vdw_parameters_SI=dict(hamaker_J=3.028e-20,h0_m=4e-10),
        bpm_parameters_SI=dict(kr=10,ks=4,kt=1e-13,kb=1e-13,gamma_normal=1e-8,gamma_shear=1e-8,gamma_twist=1e-22,gamma_bend=1e-22))
    (a.output/'provenance.json').write_text(json.dumps(note,indent=2))
    rows=[]
    for model in ('bpm_strict','bpm_near','vdw','vdw_half_dt'):
        rows+=run_case(a.output,xyz,r,rho,model,str(a.plugin.resolve()),a.lammps,2.5e-5 if model=='vdw_half_dt' else 5e-5)
        with (a.output/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Original-case BPM–vdW diagnostic','','The two uploaded coordinate sets are numerically identical. This is a new controlled relaxation experiment, not a replay of a proven historical trajectory.','','| Model | Time (us) | Bonds | Touching pairs | Contact components | Rg/d | Exposure | RMS displacement/d | KE (fJ) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        if x['step']==0:continue
        lines.append(f"| {x['model']} | {x['time_us']:g} | {x['permanent_bonds']} | {x['contacts']} | {x['contact_components']} | {x['rg_over_d']:.6f} | {x['mean_exposure']:.6f} | {x['aligned_rms_over_d']:.6g} | {x['kinetic_fJ']:.6g} |")
    lines+=['','BPM strict has only actual near-touching bonds; BPM near adds bonds across initial gaps up to 0.5% diameter, deliberately and explicitly. Neither moves the input coordinates. vdW has no permanent bonds.','The half-timestep vdW run checks numerical sensitivity at identical physical duration. A completed run is not proof of mechanical equilibrium; inspect kinetic energy and midpoint-to-final changes.','These one-case results do not establish the cause of a dataset-wide ML score difference.']
    (a.output/'results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
