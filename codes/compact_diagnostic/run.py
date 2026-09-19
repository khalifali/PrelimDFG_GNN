"""Diagnostic only: unchanged production estimator and generator."""
import argparse,csv,json,sys,subprocess,hashlib,time
from pathlib import Path
import numpy as np
from scipy.stats import linregress
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'structure'),str(Path(__file__).resolve().parents[1]/'generation')]
from descriptors import measure,count_boxes
from tunable import generate,rg2
from generate import inspect
def write(p,rows):
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def one(a):
    p=generate(a.n,a.df,a.kf,np.random.default_rng(a.seed))
    m,e=inspect(p);v,c=measure(p,np.ones(len(p)))
    a.output.mkdir(parents=True)
    np.savetxt(a.output/'particles.csv',np.c_[p*.5e-6,np.full(len(p),.5e-6)],delimiter=',',header='x_m,y_m,z_m,radius_m',comments='')
    np.savetxt(a.output/'contacts.csv',e+1,delimiter=',',header='id_i,id_j',comments='',fmt='%d')
    (a.output/'result.json').write_text(json.dumps(dict(**m,**v,rg2=rg2(p))))
def run(a):
    a.output.mkdir(parents=True,exist_ok=False);curves=[];fits=[];production=[]
    rng=np.random.default_rng(9192026);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    # Cubes test finite-size convergence; chains test an open limiting geometry.
    for kind,size in [('cube',5),('cube',10),('cube',20),('cube',30),('chain',100),('chain',1000)]:
        p=np.indices((size,size,size)).reshape(3,-1).T.astype(float) if kind=='cube' else np.c_[np.arange(size),np.zeros((size,2))]
        p-=p.mean(0);label=f'{kind}_{len(p)}';L=np.ptp(p,axis=0).max()+1
        for orient,q in [('aligned',np.eye(3)),('rotated',rotation)]:
            pp=p@q
            v,_=measure(pp,np.full(len(p),.5));production.append(dict(reference=label,orientation=orient,**v))
            for lower,upper in [(1,L/4),(2,L/4),(1,L/8),(2,L/8)]:
                if upper<=lower:continue
                sizes=np.geomspace(lower,upper,12)
                for shift in (0.,.25,.5,.75):
                    counts=[count_boxes(pp,.5,s,pp.min(0)-.5-shift*s) for s in sizes]
                    fit=linregress(-np.log(sizes),np.log(counts))
                    fits.append(dict(reference=label,orientation=orient,lower=lower,upper=upper,shift=shift,slope=fit.slope,r2=fit.rvalue**2,span_decades=np.log10(upper/lower)))
                    curves.extend(dict(reference=label,orientation=orient,lower=lower,upper=upper,shift=shift,size=s,count=int(c)) for s,c in zip(sizes,counts))
        write(a.output/'reference_production.csv',production);write(a.output/'window_fits.csv',fits);write(a.output/'window_curves.csv',curves)
        print('Reference complete',label,flush=True)
    # Exact necessary condition at the mandatory dimer -> trimer step.
    # Internal radius=1, dimer Rg²=1.6. Centre of new tangent sphere can be at most 3
    # from the dimer midpoint, giving target Rg² <= 0.6+2/3+2 = 3.266666...
    settings=[(2.6,.8),(2.8,.6),(2.8,.7),(2.8,.8),(2.9,.6),(2.9,.7),(2.9,.8),(2.9,1.)]
    constraints=[]
    for df,kf in settings:
        target=(3/kf)**(2/df);d2=4.5*(target-(2/3)*1.6-(1/3)*.6)
        constraints.append(dict(df=df,kf=kf,trimer_translation_squared=d2,trimer_lower_bound=3.,trimer_upper_bound=9.,necessary_bounds_pass=bool(3-1e-12<=d2<=9+1e-12)))
    write(a.output/'trimer_constraints.csv',constraints)
    attempts=[]
    for n in (25,100,1000):
        for df,kf in settings:
            case=f'N{n}_df{df}_kf{kf}';seed=int.from_bytes(hashlib.sha256(('compact-diagnostic:'+case).encode()).digest()[:4],'little')
            folder=a.output/'candidates'/case;log=a.output/(case+'.log')
            start=time.monotonic();status='failed';error=''
            with log.open('w') as f:
                try:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'one','--output',str(folder),'--n',str(n),'--df',str(df),'--kf',str(kf),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
                    status='generated'
                except subprocess.TimeoutExpired:error='60-second diagnostic budget'
                except subprocess.CalledProcessError:error=log.read_text()[-1000:]
            row=dict(case=case,n=n,df=df,kf=kf,seed=seed,status=status,error=error,seconds=time.monotonic()-start)
            if status=='generated':row.update(json.loads((folder/'result.json').read_text()))
            attempts.append(row);write(a.output/'generation_attempts.csv',attempts)
            print(case,status,flush=True)
    lines=['# Compactness diagnostic','',
      'Production descriptors and generator are unchanged. Reference results compare fixed fitting windows and grid shifts; no window is selected by proximity to a desired slope. Finite arrays have finite-scale exponents; no finite sphere assembly has an asymptotic dimension of 1 at all scales.',
      '',
      '24 bounded generation attempts: eight coupled settings, N=25,100,1000, one seed each. A timeout or exhausted search is not mathematical impossibility. Successful candidates are unrelaxed diagnostics, not accepted training additions.',
      '',
      f'Generated {sum(r["status"]=="generated" for r in attempts)}/{len(attempts)} candidates.',
      '',
      'See reference_production.csv, window_fits.csv and window_curves.csv for all reference data. trimer_constraints.csv records only a necessary geometric bound at the earliest construction step, not a sufficient feasibility condition. generation_attempts.csv retains errors and measurements.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['one','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int);p.add_argument('--df',type=float);p.add_argument('--kf',type=float);p.add_argument('--seed',type=int);a=p.parse_args();globals()[a.phase](a)
