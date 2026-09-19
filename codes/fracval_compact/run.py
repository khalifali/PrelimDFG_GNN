"""Compactness trial using original FracVAL kernels and explicit subcluster sizes."""
import argparse, concurrent.futures, csv, hashlib, json, os, re, shutil
import subprocess, sys, threading, time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'structure'))
from descriptors import measure

ORDER=['Ctes','random','a_Random_PP','RAND_SAMPLE','PCA_cca','PCA_Subclusters_module','Save_results_CC','CCA_module','Frac_VAL_CCA']
FLAGS=['-O2','-ffree-line-length-none','-fdefault-real-8','-fdefault-double-8']

def generate(work,seconds):
    """Drain stdout continuously, retaining <=1 MB while counting all restarts."""
    counts={'pc_restarts':0,'cc_restarts':0,'log_bytes_seen':0}
    proc=subprocess.Popen(['./fracval'],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        env={**os.environ,'GFORTRAN_UNBUFFERED_ALL':'y'})
    def drain():
        with (work/'generation.log').open('wb') as f:
            kept=0
            for line in proc.stdout:
                counts['log_bytes_seen']+=len(line)
                counts['pc_restarts']+=int(b'(PC not able' in line)
                counts['cc_restarts']+=int(b'(CC not able' in line)
                if kept<1000000:
                    chunk=line[:1000000-kept];f.write(chunk);kept+=len(chunk)
    thread=threading.Thread(target=drain);thread.start()
    start=time.monotonic();timed_out=False
    try:proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out=True;proc.kill();proc.wait()
    finally:thread.join();proc.stdout.close()
    return dict(**counts,elapsed_seconds=time.monotonic()-start,returncode=proc.returncode,timed_out=timed_out)

def write_geometry(work,data):
    n=len(data);si=data*1e-6
    np.savetxt(work/'particles.csv',data,delimiter=',',header='x_over_d,y_over_d,z_over_d,radius_over_d',comments='')
    with (work/'post_000000000000.txt').open('w') as f:
        f.write(f'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{n}\nITEM: BOX BOUNDS ff ff ff\n')
        for k in range(3):f.write(f'{si[:,k].min()-1e-6:.16g} {si[:,k].max()+1e-6:.16g}\n')
        f.write('ITEM: ATOMS id type x y z radius\n')
        for i,a in enumerate(si):f.write(f'{i+1} 1 '+' '.join(f'{v:.16g}' for v in a)+'\n')
    xyz=' '.join(f'{v:.16g}' for v in si[:,:3].ravel());r=' '.join(f'{v:.16g}' for v in si[:,3])
    ids=' '.join(map(str,range(n)));ends=' '.join(map(str,range(1,n+1)))
    (work/'particles.vtp').write_text(f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData><Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0"><PointData Scalars="radius"><DataArray type="Float64" Name="radius" format="ascii">{r}</DataArray></PointData><Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{xyz}</DataArray></Points><Verts><DataArray type="Int64" Name="connectivity" format="ascii">{ids}</DataArray><DataArray type="Int64" Name="offsets" format="ascii">{ends}</DataArray></Verts></Piece></PolyData></VTKFile>''')

def attempt(task,src,out,seconds):
    df,kf,seed,subcluster=task
    name=f'Df{df}_kf{kf}_seed{seed}_sub{subcluster or 50}'
    work=out/name;work.mkdir(exist_ok=False)
    for p in src.glob('*.f90'):shutil.copy2(p,work/p.name)
    row=dict(name=name,n=1000,requested_df=df,kf=kf,seed=seed,subcluster_size=subcluster or 50,
        configuration='explicit_subcluster_size' if subcluster else 'original_default',time_limit_seconds=seconds)
    p=work/'Ctes.f90';s=p.read_text()
    for key,value in dict(N=1000,Df=df,kf=kf,rp_g=.5,rp_gstd=1.,Quantity_aggregates=1,Ext_case=1).items():
        s,count=re.subn(r'(\b'+key+r'\s*=)[^!\n]+',lambda m:m[1]+str(value)+' ',s,flags=re.I)
        assert count==1,(key,count)
    p.write_text(s)
    if subcluster:
        # The release hardcodes 50 for N>500, ignoring Nsubcl_perc. Change only
        # that configuration branch in a working copy; vendor files stay intact.
        assert 2<=subcluster<=1000 and 1000%subcluster==0
        p=work/'CCA_module.f90';s=p.read_text()
        s,count=re.subn(r'N_subcl= 50\s*!Number of PP in each sub-cluster',
            f'N_subcl= {subcluster} !Explicit pilot subcluster-size configuration',s)
        assert count==1; p.write_text(s)
    p=work/'Frac_VAL_CCA.f90';s=p.read_text()
    assert s.count('not_able_cca = .FALSE.')==1
    s=s.replace('not_able_cca = .FALSE.',f'integer :: seed_size, seed_i\ninteger, allocatable :: seed_values(:)\ncall random_seed(size=seed_size)\nallocate(seed_values(seed_size))\nseed_values = [( {seed} + 37*seed_i, seed_i=1,seed_size)]\ncall random_seed(put=seed_values)\nnot_able_cca = .FALSE.',1)
    p.write_text(s);(work/'RESULTS').mkdir()
    try:
        with (work/'compile.log').open('w') as f:
            subprocess.run(['gfortran',*FLAGS,*[x+'.f90' for x in ORDER],'-o','fracval'],cwd=work,
                stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
        row.update(generate(work,seconds))
        if row['timed_out']:raise TimeoutError('Generation time limit reached')
        if row['returncode']!=0:raise RuntimeError('Fortran process failed')
        data=np.loadtxt(next((work/'RESULTS').glob('*.dat')))
        assert data.shape==(1000,4) and np.isfinite(data).all() and np.allclose(data[:,3],.5)
        xyz=data[:,:3];r=data[:,3];dist=squareform(pdist(xyz));np.fill_diagonal(dist,np.inf)
        overlap=max(0.,float(1-dist.min()))
        adjacency=dist<=1+2e-6;components=int(connected_components(adjacency,directed=False,return_labels=False))
        row.update(max_overlap_over_d=overlap,components=components,contacts=int(adjacency.sum()/2),
            mean_coordination=float(adjacency.sum()/1000),mean_neighbors_within_1p5d=float((dist<=1.5).sum()/1000))
        if overlap>2e-6 or components!=1:raise ValueError('Geometry failed connectivity/overlap QA')
        desc,curves=measure(xyz,r);row.update(desc)
        rg=float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2,axis=1))+.6*.5**2))
        row.update(rg_over_d=rg,mass_radius_relative_residual=float(1000/(kf*(rg/.5)**df)-1),status='measured')
        write_geometry(work,data)
        with (work/'box_curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    except Exception as e:row.update(status='timeout' if row.get('timed_out') else 'failed',error=repr(e))
    (work/'measurement.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['original','larger'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parents[1]/'fracval'/'vendor'
    for name,digest in json.loads((src/'SHA256.json').read_text()).items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
    tasks=([(2.6,.8,919260,0),(2.8,.8,919280,0),(2.9,.9,918290,0)] if a.stage=='original'
           else [(2.8,.8,919280,500),(2.9,.9,918290,500),(2.9,1.,919291,500)])
    seconds=600 if a.stage=='original' else 900
    (a.output/'protocol.json').write_text(json.dumps(dict(tasks=tasks,seconds=seconds,flags=FLAGS,
        n=1000,stage=a.stage,selection='Report all; rank compactness by measured hull porosity, not Dbox target'),indent=2))
    (a.output/'compiler.txt').write_text(subprocess.check_output(['gfortran','--version'],text=True))
    (a.output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(attempt,t,src,a.output,seconds) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result());(a.output/'measurements.json').write_text(json.dumps(rows,indent=2))
    lines=['# FracVAL compactness trial: '+a.stage,'',
      'N=1000, monodisperse; no DEM or ML. Dbox estimator unchanged. All attempts retained.',
      'Original GPLv3 source: https://doi.org/10.17632/mgf8wdcsfb.1',
      'Larger-subcluster cases change only the hardcoded N>500 subcluster-size configuration in working copies.','',
      '| Df requested | kf | Subcluster size | Status | Hull porosity | Rg/d | Mean coordination | Neighbors within 1.5d | Measured Dbox |',
      '|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows,key=lambda x:x.get('hull_porosity',2)):
        vals=[r['requested_df'],r['kf'],r['subcluster_size'],r['status']]+[r.get(k,'—') for k in
            ['hull_porosity','rg_over_d','mean_coordination','mean_neighbors_within_1p5d','box_dimension']]
        lines.append('| '+' | '.join(str(v) for v in vals)+' |')
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
