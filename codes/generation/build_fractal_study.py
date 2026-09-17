"""Old 135-case design plus N=500: 180 independent tunable-fractal cases."""
import argparse,csv,hashlib,json,platform
from pathlib import Path
import numpy as np
from generate import inspect,write_vtk
from tunable import generate,rg2


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for diameter in (1.,1.5,2.):
        for n in (50,100,200,500):
            for df,kf,label in [(1.8,1.3,'open'),(2.2,1.1,'intermediate'),(2.6,.8,'compact')]:
                for rep in range(1,6):
                    case=f'fractal_dp{diameter:g}_N{n:04d}_Df{df:g}_kf{kf:g}_rep{rep:02d}'
                    seed=int.from_bytes(hashlib.sha256(f'18427:fractal180-v1:{case}'.encode()).digest()[:8],'little')
                    points=generate(n,df,kf,np.random.default_rng(seed))
                    metrics,edges=inspect(points)
                    target=(n/kf)**(1/df)
                    error=abs(metrics['rg_over_radius']/target-1)
                    if error>1e-9:raise RuntimeError(f'{case}: mass-radius target failed')
                    radius=diameter*.5e-6;physical=points*radius
                    folder=output/case;folder.mkdir()
                    np.savetxt(folder/'particles.csv',np.column_stack([physical,np.full(n,radius)]),delimiter=',',header='x_m,y_m,z_m,radius_m',comments='',fmt='%.17g')
                    np.savetxt(folder/'contacts.csv',edges+1,delimiter=',',header='id_i,id_j',comments='',fmt='%d')
                    write_vtk(folder/'particles.vtk',physical,radius)
                    row=dict(case=case,method='tunable_fractal',family='tunable_fractal',detector_radius_over_r='',sample=rep,replicate=rep,seed=seed,diameter_um=diameter,
                             requested_df=df,requested_kf=kf,structural_class=label,class_definition='prescribed_Df_kf_pair',
                             parameter_group=f'tunable_fractal|N{n}|Df{df}|kf{kf}',
                             study_subset='extension_N500' if n==500 else 'historical_design_core',
                             target_rg_over_radius=target,mass_radius_relative_error=error,
                             compactness_score=n**(1/3)/metrics['rg_over_radius'],**metrics)
                    (folder/'metadata.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row)
                print(f'd={diameter} um N={n} Df={df} kf={kf}: 5 passed',flush=True)
    with (output/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=dict(case_count=len(records),historical_design_core=135,extension_N500=45,
        all_geometry_checks_passed=True,maximum_mass_radius_relative_error=max(r['mass_radius_relative_error'] for r in records),
        master_seed=18427,algorithm='tunable particle-cluster seeds and hierarchical cluster-cluster merging',
        source_note='Reconstructed formulation, not recovered historical generator or identical old coordinates',
        grouping='Keep all diameters and realizations for each N/Df/kf together',
        python=platform.python_version(),numpy=np.__version__,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('generate.py','tunable.py','build_fractal_study.py')})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
