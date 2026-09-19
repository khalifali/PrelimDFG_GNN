"""Bounded target-blind tunable-generator pilot and unchanged descriptor reference checks."""
import argparse,csv,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'generation'),str(HERE.parent/'structure')]
from tunable import generate
from generate import inspect
from descriptors import measure

SETTINGS=[(1.5,1.3),(1.8,1.3),(2.2,1.1),(2.6,.8),(2.8,.8),(2.9,1.)]
def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def measurements(p):
    values,curves=measure(p,np.ones(len(p)))
    return values,curves
def one(a):
    rng=np.random.default_rng(a.seed);p=generate(a.n,a.df,a.kf,rng)
    metrics,edges=inspect(p)
    a.output.mkdir(parents=True,exist_ok=False)
    np.savetxt(a.output/'particles.csv',np.c_[p*.5e-6,np.full(len(p),.5e-6)],delimiter=',',header='x_m,y_m,z_m,radius_m',comments='',fmt='%.17g')
    np.savetxt(a.output/'contacts.csv',edges+1,delimiter=',',header='id_i,id_j',comments='',fmt='%d')
    meta=dict(case=a.output.name,seed=a.seed,method='tunable_fractal',requested_df=a.df,requested_kf=a.kf,diameter_um=1.,root_geometry_id=a.output.name,**metrics)
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
def pilot(a):
    a.output.mkdir(parents=True,exist_ok=False);geo=a.output/'geometries';geo.mkdir()
    planned=[]
    for n in (100,350,1000):
        for df,kf in SETTINGS:
            case=f'coverage_N{n:04d}_Df{df}_kf{kf}'
            seed=int.from_bytes(hashlib.sha256(('coverage-pilot-20260919:'+case).encode()).digest()[:4],'little')
            planned.append(dict(case=case,n=n,df=df,kf=kf,seed=seed))
    csvwrite(a.output/'planned.csv',planned)
    accepted=[];attempts=[];descs=[]
    for spec in planned:
        folder=geo/spec['case'];start=time.monotonic();status='failed';error=''
        cmd=[sys.executable,str(Path(__file__).resolve()),'one','--output',str(folder),'--n',str(spec['n']),'--df',str(spec['df']),'--kf',str(spec['kf']),'--seed',str(spec['seed'])]
        with (a.output/(spec['case']+'.log')).open('w') as f:
            try:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
                status='generated'
            except subprocess.TimeoutExpired:error='180-second generation budget exceeded'
            except subprocess.CalledProcessError as e:error=str(e)
        if status=='generated':
            meta=json.loads((folder/'metadata.json').read_text());accepted.append(meta)
            arr=np.loadtxt(folder/'particles.csv',delimiter=',',skiprows=1)
            try:
                v,curve=measure(arr[:,:3],arr[:,3]);descs.append(dict(case=spec['case'],stage='initial',n=spec['n'],**v))
                csvwrite(folder/'box_curves_initial.csv',curve)
            except ValueError as e:descs.append(dict(case=spec['case'],stage='initial',n=spec['n'],measurement_error=str(e)))
        attempts.append(dict(**spec,status=status,error=error,seconds=time.monotonic()-start))
        csvwrite(a.output/'attempts.csv',attempts);csvwrite(a.output/'initial_descriptors.csv',descs)
        print(attempts[-1],flush=True)
    csvwrite(geo/'manifest.csv',accepted)
    refs=[]
    for kind,sizes in [('line',(50,200,1000)),('simple_cubic',(5,10,20))]:
        for size in sizes:
            p=(np.c_[np.arange(size)*2,np.zeros((size,2))] if kind=='line' else np.indices((size,size,size)).reshape(3,-1).T*2.)
            v,curve=measurements(p)
            refs.append(dict(reference=kind,n=len(p),**v))
            csvwrite(a.output/f'reference_{kind}_{len(p)}_curves.csv',curve)
    csvwrite(a.output/'reference_descriptors.csv',refs)
    (a.output/'generation_summary.json').write_text(json.dumps(dict(planned=len(planned),generated=len(accepted),references_only='Line and simple-cubic structures are estimator diagnostics, not training cases; finite-scale slopes need not equal continuum asymptotic dimensions.'),indent=2))
    report(a)
    if not accepted:raise RuntimeError('No candidate generated; inspect attempts')
def final(a):
    rows=[]
    for folder in sorted((a.output/'bonded').glob('*')):
        if not (folder/'case.json').exists():continue
        assessment=json.loads((folder/'assessment.json').read_text()) if (folder/'assessment.json').exists() else {}
        r=dict(case=folder.name,mechanical_passed=assessment.get('passed',False))
        if (folder/'final_particles.csv').exists():
            arr=np.loadtxt(folder/'final_particles.csv',delimiter=',',skiprows=1)
            try:
                v,curve=measure(arr[:,:3],arr[:,3]);r.update(n=len(arr),**v)
                csvwrite(folder/'box_curves_final.csv',curve)
            except ValueError as e:r['measurement_error']=str(e)
        else:r['measurement_error']='No final geometry'
        rows.append(r)
    csvwrite(a.output/'final_descriptors.csv',rows);report(a)
def report(a):
    def read(name):
        p=a.output/name
        return list(csv.DictReader(p.open())) if p.exists() else []
    lines=['# Coverage feasibility pilot','',
        '18 planned independent tunable-generator attempts: N=100,350,1000 and six coupled requested Df/kf settings; one deterministic realization each. 180-second budget per attempt. No ML or exposure selection. FracVAL remains paused.','',
        'Initial generated cases are candidates only. Final coverage requires successful BPM assessment and interpretable measured descriptors. Failed attempts and measurements remain recorded.','',
        'The box-counting estimator is unchanged. References test finite-size behaviour, not a requirement that finite-window slopes equal 1 or 3. No automatic correction or extrapolation of Dbox is applied.']
    for title,file in [('Initial candidates','initial_descriptors.csv'),('Reference geometries','reference_descriptors.csv'),('Final BPM candidates','final_descriptors.csv')]:
        lines+=['','## '+title,'','| Case | N | Dbox | Porosity | Quality / error | Mechanical pass |','|---|---:|---:|---:|---|---|']
        for r in read(file):
            lines.append('| '+' | '.join(str(r.get(k,'')) for k in [('reference' if 'reference' in r else 'case'),'n','box_dimension','hull_porosity',('measurement_error' if r.get('measurement_error') else 'box_quality'),'mechanical_passed'])+' |')
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['one','pilot','final']);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int);p.add_argument('--df',type=float);p.add_argument('--kf',type=float);p.add_argument('--seed',type=int);a=p.parse_args();globals()[a.phase](a)
