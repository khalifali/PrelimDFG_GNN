#!/usr/bin/env python3
"""Prepare native LAMMPS BPM cases from all rows of a geometry manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation'))
from generate import inspect

PARAMETERS = dict(density_kg_m3=1050.,kr_N_m=10.,ks_N_m=4.,
                  kt_Nm=1e-13,kb_Nm=1e-13,gamma_normal_kg_s=1e-8,
                  gamma_shear_kg_s=1e-8,gamma_twist_kg_m2_s=1e-22,
                  gamma_bend_kg_m2_s=1e-22,young_Pa=1e9,poisson=.3,
                  restitution=.5,friction=.3,numerical_drag_kg_s=1e-8,numerical_rot_drag_kg_m2_s=1e-22,dt_s=1e-10,steps=20000,
                  dump_every=10000,kick_velocity_m_s=.001,kick_omega_s=1000.)


def write_initial(path, xyz, radii, bounds):
    """Native dump in MICRO units, compatible with the LAMFOAM converter."""
    with path.open('w') as f:
        f.write('ITEM: TIME\n0\nITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n')
        f.write(f'{len(xyz)}\nITEM: BOX BOUNDS ff ff ff\n')
        np.savetxt(f,bounds,fmt='%.17g')
        f.write('ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz radius fx fy fz\n')
        for i,(p,r) in enumerate(zip(xyz,radii),1):
            f.write(f'{i} 1 '+' '.join(f'{x:.17g}' for x in p)+f' 0 0 0 0 0 0 {r:.17g} 0 0 0\n')


def prepare_case(source, dest, params, metadata):
    a=np.loadtxt(source/'particles.csv',delimiter=',',skiprows=1,ndmin=2)
    xyz=a[:,:3]*1e6; radii=a[:,3]*1e6; n=len(a)
    if not np.allclose(radii,radii[0],rtol=1e-12,atol=0):
        raise ValueError('This initial campaign assumes equal-sized particles.')
    _,expected=inspect(xyz/radii[0])
    edges=np.loadtxt(source/'contacts.csv',delimiter=',',skiprows=1,dtype=int,ndmin=2)
    if not np.array_equal(expected+1,edges):
        raise ValueError('Contact file differs from independently checked geometry.')
    dest.mkdir(parents=True,exist_ok=False)
    for part in ['post','bond','generated']:(dest/part).mkdir()
    bounds=np.column_stack([xyz.min(axis=0)-10*radii[0],xyz.max(axis=0)+10*radii[0]])
    rho=params['density_kg_m3']*1e-3
    with (dest/'particles.data').open('w') as f:
        f.write(f'Generated aggregate, MICRO units\n\n{n} atoms\n{len(edges)} bonds\n\n1 atom types\n1 bond types\n\n')
        for (lo,hi),ax in zip(bounds,'xyz'):f.write(f'{lo:.17g} {hi:.17g} {ax}lo {ax}hi\n')
        f.write('\nAtoms # bpm/sphere\n\n')
        for i,(p,r) in enumerate(zip(xyz,radii),1):
            f.write(f'{i} 1 1 {2*r:.17g} {rho:.17g} '+' '.join(f'{x:.17g}' for x in p)+'\n')
        f.write('\nBonds\n\n')
        for i,(u,v) in enumerate(edges,1):f.write(f'{i} 1 {u} {v}\n')
    # Store the exact generated geometry separately from the perturbed DEM series.
    write_initial(dest/'generated'/'post_0.txt',xyz,radii,bounds)
    rng=np.random.default_rng(int(metadata['seed']))
    v=rng.normal(size=(n,3))*params['kick_velocity_m_s']
    omega=rng.normal(size=(n,3))*params['kick_omega_s']*1e-6
    mass=rho*4*np.pi*radii**3/3; inertia=.4*mass*radii**2
    v-=np.average(v,axis=0,weights=mass)
    centered=xyz-np.average(xyz,axis=0,weights=mass)
    tensor=sum(m*(np.dot(p,p)*np.eye(3)-np.outer(p,p)) for m,p in zip(mass,centered))
    tensor+=np.eye(3)*inertia.sum()
    angular=np.sum(np.cross(centered,mass[:,None]*v)+inertia[:,None]*omega,axis=0)
    spin=np.linalg.solve(tensor,angular)
    v-=np.cross(spin,centered);omega-=spin
    with (dest/'kick.in').open('w') as f:
        f.write('# Small deterministic disturbance; zero total linear/angular momentum. MICRO units.\n')
        for i,(vel,w) in enumerate(zip(v,omega),1):
            f.write(f'set atom {i} vx {vel[0]:.17g} vy {vel[1]:.17g} vz {vel[2]:.17g}\n')
            f.write(f'set atom {i} omega {w[0]:.17g} {w[1]:.17g} {w[2]:.17g}\n')
    p=params;diam=2*radii[0]
    coeff=[p['kr_N_m']*1e3,p['ks_N_m']*1e3,p['kt_Nm']*1e15,p['kb_Nm']*1e15,
           1.,1.,1.,1.,p['gamma_normal_kg_s']*1e9,p['gamma_shear_kg_s']*1e9,
           p['gamma_twist_kg_m2_s']*1e21,p['gamma_bend_kg_m2_s']*1e21]
    script=f'''# Native LAMMPS, no LIGGGHTS or LAMFOAM plugin needed.
# Run from this case directory: lmp -in in.lammps
# MICRO: length um, time us, mass pg, force nN, energy fJ.
units micro
atom_style bpm/sphere
atom_modify map array
boundary f f f
newton on off
read_data particles.data extra/special/per/atom {n}
comm_modify vel yes cutoff {3*diam:.17g}
neighbor {0.2*diam:.17g} bin
neigh_modify every 1 delay 0 check yes
# Direct bonded neighbours use BPM only; other pairs retain repulsive contacts.
special_bonds lj 0 1 1 coul 1 1 1
pair_style granular {1.2*diam:.17g}
pair_coeff * * hertz/material {p['young_Pa']*.001:.17g} {p['restitution']} {p['poisson']} tangential mindlin NULL 1.0 {p['friction']} damping tsuji limit_damping
# This permanent cohesive bond transmits normal/shear forces and bend/twist torques.
# No vdW. Nonbreaking bonds: failure coefficients are placeholders, ignored.
# Explicit compatibility settings are selected by prepare.py for installed version.
bond_style bpm/rotational break no smooth no __BPM_COMPAT__
bond_coeff 1 {' '.join(f'{x:.17g}' for x in coeff)}
fix integr all nve/bpm/sphere
# Numerical energy removal, not a calibrated surrounding-fluid model.
fix damp_translation all viscous {p['numerical_drag_kg_s']*1e9:.17g}
fix damp_rotation all viscous/sphere {p['numerical_rot_drag_kg_m2_s']*1e21:.17g}
# No gravity, surrounding fluid, walls or compression. Free aggregate.
timestep {p['dt_s']*1e6:.17g}
compute kt all ke
compute kr all erotate/sphere
variable kinetic equal c_kt+c_kr
thermo {p['dump_every']}
thermo_style custom step time atoms bonds c_kt c_kr v_kinetic
thermo_modify lost error format float %.16e
# Initialise bond reference lengths and orientations BEFORE the velocity kick.
run 0
include kick.in
# Particle snapshots directly accepted by posttovtk.py (both VTK and VTP/PVD).
dump particles all custom {p['dump_every']} post/post_*.txt id type x y z vx vy vz omegax omegay omegaz radius fx fy fz
 dump_modify particles sort id format float %.16e time yes
compute bondIDs all property/local batom1 batom2
compute bondForces all bond/local dist force b5 b6 b7
dump bonds all local {p['dump_every']} bond/bond_*.txt c_bondIDs[1] c_bondIDs[2] c_bondForces[*]
dump_modify bonds format float %.16e time yes
run {p['steps']}
# Only this binary preserves the BPM reference state for continuation.
write_restart final.restart
write_data final.data
'''
    (dest/'in.template').write_text(script)
    # Old stable LAMMPS uses the particle frame and DEM damping already.
    (dest/'in.lammps').write_text(script.replace('__BPM_COMPAT__',''))
    record=dict(source_geometry=metadata,parameters_SI=params,
                units=dict(length='micrometre',time='microsecond',mass='picogram',force='nanonewton'),
                input_particles_sha256=hashlib.sha256((source/'particles.csv').read_bytes()).hexdigest(),
                bond_model='bpm/rotational; permanent; legacy particle frame / DEM damping',
                parameter_status='illustrative numerical verification; not material calibration')
    (dest/'case.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int);parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--dt',type=float,default=1e-10)
    parser.add_argument('--no-kick',action='store_true',help='Stationary control, not a stability test')
    args=parser.parse_args()
    if args.steps<=0 or not np.isfinite(args.dt) or args.dt<=0:parser.error('Positive steps and finite positive dt required')
    if args.steps%2:parser.error('Steps must be even for initial, midpoint, and final dumps')
    with (args.geometry/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    if args.limit is not None:rows=rows[:args.limit]
    args.output.mkdir(parents=True,exist_ok=False)
    p=dict(PARAMETERS,steps=args.steps,dt_s=args.dt,dump_every=args.steps//2)
    if args.no_kick:p.update(kick_velocity_m_s=0.,kick_omega_s=0.)
    for row in rows:prepare_case(args.geometry/row['case'],args.output/row['case'],p,row)
    (args.output/'campaign.json').write_text(json.dumps(dict(cases=[r['case'] for r in rows],count=len(rows),parameters_SI=p),indent=2)+'\n')
    print(f'Prepared {len(rows)} bonded LAMMPS cases in {args.output}')

if __name__=='__main__':main()
