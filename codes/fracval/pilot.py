"""Bounded FracVAL geometry calibration; no DEM or ML, no estimator tuning."""
import argparse, concurrent.futures, csv, hashlib, io, json, re, shutil
import subprocess, sys, tarfile, urllib.request
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'structure'))
from descriptors import measure

URL='https://data.mendeley.com/public-files/datasets/mgf8wdcsfb/files/cfc00699-9b22-4b5c-8b2e-06fd452024e9/file_downloaded'
SHA='7c743d920a211257468b7d45d1a9a15ca21d2edc3647e87a1564de3ce63457d2'
ORDER=['Ctes','random','a_Random_PP','RAND_SAMPLE','PCA_cca','PCA_Subclusters_module','Save_results_CC','CCA_module','Frac_VAL_CCA']

def attempt(task,src,out,seconds):
    target,df,kf,seed=task
    name=f'target{target}_Df{df}_kf{kf}_seed{seed}'
    work=out/name;work.mkdir(exist_ok=True)
    for p in src.glob('*.f90'):shutil.copy2(p,work/p.name)
    p=work/'Ctes.f90';s=p.read_text()
    for key,value in dict(N=1000,Df=df,kf=kf,rp_g=.5,rp_gstd=1.,Quantity_aggregates=1,Ext_case=1).items():
        s,n=re.subn(r'(\b'+key+r'\s*=)[^!\n]+',lambda m:m[1]+str(value)+' ',s,flags=re.I)
        assert n==1,(key,n)
    p.write_text(s)
    # Seed initialization only: preserve the published aggregation algorithm.
    p=work/'Frac_VAL_CCA.f90';s=p.read_text()
    s=s.replace('not_able_cca = .FALSE.',f'integer :: seed_size, seed_i\ninteger, allocatable :: seed_values(:)\ncall random_seed(size=seed_size)\nallocate(seed_values(seed_size))\nseed_values = [( {seed} + 37*seed_i, seed_i=1,seed_size)]\ncall random_seed(put=seed_values)\nnot_able_cca = .FALSE.',1)
    p.write_text(s);(work/'RESULTS').mkdir(exist_ok=True)
    row=dict(name=name,target_box_dimension=target,requested_df=df,kf=kf,seed=seed,n=1000)
    try:
        cmd=['gfortran','-O2','-ffree-line-length-none','-fdefault-real-8','-fdefault-double-8']+[x+'.f90' for x in ORDER]+['-o','fracval']
        with (work/'compile.log').open('w') as f:subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
        with (work/'generation.log').open('w') as f:subprocess.run(['./fracval'],cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=seconds)
        data=np.loadtxt(next((work/'RESULTS').glob('*.dat')))
        assert data.shape==(1000,4) and np.isfinite(data).all()
        xyz=data[:,:3];r=data[:,3];d=2*r[0]
        dist=squareform(pdist(xyz));np.fill_diagonal(dist,np.inf)
        overlap=max(0.,float(1-dist.min()/d))
        adjacency=dist<=d*(1+2e-6)
        components=connected_components(adjacency,directed=False,return_labels=False)
        row.update(max_overlap_over_d=overlap,components=int(components),contacts=int(adjacency.sum()/2))
        if overlap>2e-6 or components!=1:raise ValueError('Geometry failed connectivity/overlap QA')
        desc,curves=measure(xyz,r);row.update(desc)
        rg=np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2,axis=1))+.6*r[0]**2)
        row['mass_radius_relative_residual']=float(1000/(kf*(rg/r[0])**df)-1)
        row['absolute_target_error']=abs(desc['box_dimension']-target)
        row['status']='measured'
        np.savetxt(work/'particles.csv',data,delimiter=',',header='x_over_d,y_over_d,z_over_d,radius_over_d',comments='')
        # Standard particle dump: choose d=1 micrometre for viewing / later DEM.
        si=data*1e-6
        with (work/'post_000000000000.txt').open('w') as f:
            f.write('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1000\nITEM: BOX BOUNDS ff ff ff\n')
            for k in range(3):f.write(f'{si[:,k].min()-1e-6:.16g} {si[:,k].max()+1e-6:.16g}\n')
            f.write('ITEM: ATOMS id type x y z radius\n')
            for i,a in enumerate(si):f.write(f'{i+1} 1 '+' '.join(f'{v:.16g}' for v in a)+'\n')
        with (work/'box_curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    except Exception as e:row.update(status='failed',error=repr(e))
    (work/'measurement.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=120);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parent/'vendor'
    manifest=json.loads((src/'SHA256.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
    # Calibration candidates are disclosed in full; targets refer to measured Dbox.
    # A match requires absolute error <=0.05; quality flags are reported separately.
    tasks=[(1.5,1.5,1.5,918150),(2.,2.,1.2,918200),(2.9,2.9,.5,918290)]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda t:attempt(t,src,a.output,a.seconds),tasks):rows.append(row)
    (a.output/'measurements.json').write_text(json.dumps(rows,indent=2))
    lines=['# FracVAL pilot: requested versus measured dimensions','',
      'Original release DOI: 10.17632/mgf8wdcsfb.1 (GPLv3). N=1000, monodisperse. No DEM relaxation or ML.',
      'Original aggregation equations retained. Changes: input constants, explicit random seed, double-precision compiler flags.',
      'Box estimator is unchanged from codes/structure/descriptors.py: 8 sizes d to L/4, 3 orientations, 2 grid shifts.',
      'Target tolerance: ±0.05. A requested mass-radius dimension is not a measured box dimension. Failed attempts are retained.','',
      '| Target Dbox | Requested Df | kf | Measured Dbox | Hull porosity | Status / quality |',
      '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['target_box_dimension']} | {r['requested_df']} | {r['kf']} | {r.get('box_dimension','—')} | {r.get('hull_porosity','—')} | {r.get('box_quality',r['status'])} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
