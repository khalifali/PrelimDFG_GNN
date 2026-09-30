#!/usr/bin/env python3
"""Export cumulative uptake curves; keep absolute CFD time and uptake time distinct."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyse(root):
    root=Path(root)
    manifest=json.loads((root/'run.json').read_text())
    settings=manifest['settings']; warm=manifest['warmup_end_s']; dt=settings['delta_t_s']
    n=manifest['particle_count']; end=manifest['end_time_s']; tolerance=dt*1e-7
    text=(root/'CFD/log.solver').read_text()
    if not re.search(r'^End\s*$',text,re.M):
        raise ValueError('Solver did not reach a normal end; refusing a complete-curve export')
    balances=[list(map(float,row.split())) for row in re.findall(r'^Transport oxygen balance (.*)$',text,re.M)]
    if any(len(row)!=5 or not all(math.isfinite(v) for v in row) for row in balances):
        raise ValueError('Invalid oxygen balance log')
    active=[r for r in balances if r[0]>=warm-tolerance]
    if len(active)!=round(settings['uptake_duration_s']/dt)+1 or abs(active[-1][0]-end)>tolerance:
        raise ValueError('Incomplete uptake history')
    if warm:
        warmsteps=re.findall(r'^LAMFOAM flow warmup step (.*)$',text,re.M)
        if len(warmsteps)!=round(warm/dt) or any('unchanged true' not in row for row in warmsteps):
            raise ValueError('Warm-up invariance not verified')
        if 'LAMFOAM flow warmup complete' not in text:
            raise ValueError('No warm-up transition')
    if any(r[2]!=0 for r in balances if r[0]<=warm+tolerance):
        raise ValueError('Particles took up oxygen before activation')
    minima=[float(v) for v in re.findall(r'^Transport oxygen marker minimum (.*)$',text,re.M)]
    if len(minima)!=len(active)-1 or min(minima)<-1e-12*settings['initial_concentration_mol_m3']:
        raise ValueError('Missing positivity diagnostics or negative concentration')
    curve=[]
    for i,(t,fluid,particle,total,error) in enumerate(active):
        if abs(t-(warm+i*dt))>tolerance or particle<0 or error>1e-8:
            raise ValueError('Invalid curve time, inventory or oxygen budget')
        if i and particle<active[i-1][2]-1e-12*max(particle,1e-30):
            raise ValueError('Cumulative absorbing-particle uptake decreased')
        curve.append(dict(cfd_time_s=t,uptake_time_s=max(0.,t-warm),particle_count=n,
                          mean_uptake_mol_per_particle=particle/n,total_uptake_mol=particle,
                          fluid_inventory_mol=fluid,relative_balance_error=error))
    with (root/'particles.csv').open() as f:
        geometry={int(r['id']):[float(r[k]) for k in ('x_m','y_m','z_m')] for r in csv.DictReader(f)}
    reference={round(r[0]/dt):r for r in active}
    individual=[]; moments=[]; snapshots=0
    for path in sorted((root/'DEM/oxygen').glob('particles.*.lammpstrj'), key=lambda p:int(p.name.split('.')[1])):
        step=int(path.name.split('.')[1]); t=step*dt
        lines=path.read_text().splitlines(); index=next(i for i,l in enumerate(lines) if l.startswith('ITEM: ATOMS '))
        columns=lines[index].split()[2:]
        if int(lines[1])!=0:
            raise ValueError('Fixed-particle tutorial advanced DEM timesteps')
        rows=[dict(zip(columns,line.split())) for line in lines[index+1:] if line.strip()]
        if len(rows)!=n or {int(r['id']) for r in rows}!=set(geometry):
            raise ValueError('Particle IDs/count changed')
        amounts=[]
        for row in rows:
            ident=int(row['id']); amount=float(row['c_lamfoamInventory_oxygen'])
            xyz=[float(row[k]) for k in ('x','y','z')]
            if any(abs(a-b)>1e-12*max(manifest['box_lengths_m']) for a,b in zip(xyz,geometry[ident])):
                raise ValueError('Fixed particle moved')
            if not math.isfinite(amount) or amount<0 or (t<=warm+tolerance and amount!=0):
                raise ValueError('Invalid inventory or uptake during warm-up')
            amounts.append(amount)
            if t>=warm-tolerance:
                individual.append(dict(cfd_time_s=t,uptake_time_s=max(0.,t-warm),particle_id=ident,uptake_mol=amount))
        if t>=warm-tolerance:
            expected=reference[step][2]
            if abs(sum(amounts)-expected)>1e-10*max(expected,1e-30):
                raise ValueError('Particle output differs from global inventory')
            moments.append(dict(cfd_time_s=t,uptake_time_s=max(0.,t-warm),
                mean_uptake_mol_per_particle=statistics.mean(amounts),
                population_variance_mol2=statistics.pvariance(amounts),
                minimum_uptake_mol=min(amounts),maximum_uptake_mol=max(amounts)))
        snapshots+=1
    if snapshots!=round(end/dt)//settings['write_interval_steps']+1:
        raise ValueError('Incomplete particle snapshots')
    out=root/'postProcessing'; out.mkdir(exist_ok=True)
    write_csv(out/'uptake_mean.csv',curve)
    write_csv(out/'uptake_particles.csv',individual)
    write_csv(out/'uptake_statistics.csv',moments)
    summary=dict(status='runtime_pass',accuracy_certified=False,particle_count=n,
        curve_rows=len(curve),particle_snapshots=snapshots,
        final_mean_uptake_mol_per_particle=curve[-1]['mean_uptake_mol_per_particle'],
        final_total_uptake_mol=curve[-1]['total_uptake_mol'],
        max_relative_balance_error=max(r[-1] for r in active),minimum_concentration=min(minima))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('case',type=Path);args=ap.parse_args()
    print(json.dumps(analyse(args.case.resolve()),indent=2))
