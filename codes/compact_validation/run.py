"""Bounded v1/v2 generation comparison and candidate estimator validation."""
import argparse, json, sys, time, subprocess, hashlib
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/p) for p in ('generation','structure')]
from tunable import generate, generate_bounded, rg2
from generate import inspect
from descriptors import box_dimension, hull_porosity
from box_ensemble import box_ensemble


def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def one(a):
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'events.jsonl').open('w',buffering=1) as f:
        def event(e): f.write(json.dumps(e)+'\n')
        rng=np.random.default_rng(a.seed)
        p=generate(a.n,a.df,a.kf,rng) if a.method=='v1' else generate_bounded(a.n,a.df,a.kf,rng,seconds=60,event=event)
    m,e=inspect(p)
    assert np.isclose(rg2(p),(a.n/a.kf)**(2/a.df),rtol=1e-10)
    np.save(a.output/'points.npy',p)
    np.savetxt(a.output/'particles.csv',np.c_[p*.5e-6,np.full(len(p),.5e-6)],delimiter=',',header='x_m,y_m,z_m,radius_m',comments='')
    np.savetxt(a.output/'contacts.csv',e+1,delimiter=',',header='id_i,id_j',comments='',fmt='%d')
    save(a.output/'geometry.json',dict(**m,rg2=float(rg2(p))))


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    attempts=[]
    for n in (100,1000):
        for df,kf in ((2.6,.8),(2.8,.7),(2.9,.7)):
            for replicate in (0,1):
                case=f'N{n}_df{df}_kf{kf}_rep{replicate}'
                seed=int.from_bytes(hashlib.sha256(('compact-v2:'+case).encode()).digest()[:4],'little')
                for method in ('v1','v2'):
                    folder=a.output/(case+'_'+method);folder.mkdir()
                    start=time.monotonic();status='generated'
                    with (folder/'generation.log').open('w') as log:
                        try:
                            subprocess.run([sys.executable,__file__,'one','--output',str(folder),'--n',str(n),'--df',str(df),'--kf',str(kf),'--seed',str(seed),'--method',method],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=65)
                        except subprocess.TimeoutExpired:status='timeout'
                        except subprocess.CalledProcessError:status='failed'
                    row=dict(case=case,method=method,n=n,df=df,kf=kf,seed=seed,status=status,seconds=time.monotonic()-start)
                    if status=='generated':
                        p=np.load(folder/'points.npy')/2
                        v,c=box_ensemble(p)
                        save(folder/'descriptor_v2.json',dict(**v,**hull_porosity(p)))
                        save(folder/'box_curves_v2.json',c)
                    attempts.append(row);save(a.output/'attempts.json',attempts)
                    print(case,method,status,flush=True)
    references=[]
    for kind,size in [('chain',100),('chain',1000),('cube',5),('cube',10),('cube',20)]:
        p=np.indices((size,size,size)).reshape(3,-1).T.astype(float) if kind=='cube' else np.c_[np.arange(size),np.zeros((size,2))]
        p-=p.mean(0)
        for orient in range(3):
            q=np.eye(3) if orient==0 else np.linalg.qr(np.random.default_rng(900+orient).normal(size=(3,3)))[0]
            pp=p@q
            old,_=box_dimension(pp)
            for budget in (8,16):
                new,curves=box_ensemble(pp,orientations=budget)
                row=dict(reference=f'{kind}{len(p)}',orientation=orient,budget=budget,legacy=old,candidate=new)
                references.append(row)
                save(a.output/f'{kind}{len(p)}_rotation{orient}_budget{budget}_curves.json',curves)
            save(a.output/'references.json',references)
        print('Reference complete',kind,len(p),flush=True)
    # Review thresholds fixed before this run; no automatic production switch.
    checks=[]
    for label in sorted({r['reference'] for r in references}):
        rows=[r for r in references if r['reference']==label]
        full=[r['candidate']['box_dimension'] for r in rows if r['budget']==16]
        changes=[abs(rows[i]['candidate']['box_dimension']-rows[i+1]['candidate']['box_dimension']) for i in range(0,len(rows),2)]
        checks.append(dict(reference=label,rotation_range=max(full)-min(full),max_budget_change=max(changes),numerical_gate_pass=bool(max(full)-min(full)<=.03 and max(changes)<=.03)))
    save(a.output/'reference_checks.json',checks)
    save(a.output/'provenance.json',dict(generation_budget_seconds=65,internal_v2_seconds=60,notes='Same seeds/settings and external timeout for both versions. No BPM, no training. Candidate descriptor uses changed invariant fitting extent. No automatic adoption.'))
    (a.output/'report.md').write_text('# Compact validation\n\nGenerated '+str(sum(r['status']=='generated' for r in attempts))+'/'+str(len(attempts))+' attempts across both versions.\n\nInspect attempts.json and per-case failure logs; compare versions within settings. See reference_checks.json for preregistered 0.03 rotation-range and budget-change gates. Passing numerical gates does not establish a true fractal scaling regime; retain span/window/local-slope checks. All candidates require BPM before admission.\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['one','run']);parser.add_argument('--output',type=Path,required=True)
    for name,t in [('n',int),('df',float),('kf',float),('seed',int)]:parser.add_argument('--'+name,type=t)
    parser.add_argument('--method',choices=['v1','v2']);a=parser.parse_args();globals()[a.phase](a)
