#!/usr/bin/env python3
"""Select matched-N DEM geometries and prepare fixed-geometry laminar uptake cases.

Usage (from this directory, with PrelimDFG_GNN checked out alongside LAMFOAM):
    python3 build_cases.py

Or point explicitly to the source checkout:
    python3 build_cases.py --source-repo /path/to/PrelimDFG_GNN

This prepares inputs only; it never invokes OpenFOAM or LAMMPS. Case directories
are created without overwriting existing runs, so use a clean project copy to
rebuild the complete matrix.
"""
import argparse, csv, hashlib, json, math, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / 'PrelimDFG_GNN'
DATA = REPO / 'data/generation/study180_v1'
EXPOSURE = REPO / 'results/study180/exposure_summary_post.json'
sys.path.insert(0, str(HERE))
from run import prepare


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def patch_blockmesh(path):
    s = path.read_text()
    s = re.sub(r'boundary\s*\(.*?\);\s*mergePatchPairs', '''boundary
(
 xmin { type patch; faces ((0 4 7 3)); }
 xmax { type patch; faces ((1 2 6 5)); }
 ymin { type wall; faces ((0 1 5 4)); }
 ymax { type wall; faces ((3 7 6 2)); }
 zmin { type wall; faces ((0 3 2 1)); }
 zmax { type wall; faces ((4 5 6 7)); }
);
mergePatchPairs''', s, flags=re.S)
    path.write_text(s)

def set_flow_bcs(case, velocity, concentration):
    cfd = case/'CFD'
    (cfd/'0/U').write_text(f'''FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({velocity:.17g} 0 0);
boundaryField
{{
 xmin {{ type fixedValue; value uniform ({velocity:.17g} 0 0); }}
 xmax {{ type zeroGradient; }}
 ymin {{ type noSlip; }}
 ymax {{ type noSlip; }}
 zmin {{ type noSlip; }}
 zmax {{ type noSlip; }}
}}
''')
    (cfd/'0/p').write_text('''FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
 xmin { type zeroGradient; }
 xmax { type fixedValue; value uniform 0; }
 ymin { type zeroGradient; }
 ymax { type zeroGradient; }
 zmin { type zeroGradient; }
 zmax { type zeroGradient; }
}
''')
    (cfd/'0/oxygen').write_text(f'''FoamFile {{ version 2.0; format ascii; class volScalarField; object oxygen; }}
dimensions [0 -3 0 0 1 0 0];
internalField uniform {concentration:.17g};
boundaryField
{{
 xmin {{ type fixedValue; value uniform {concentration:.17g}; }}
 xmax {{ type inletOutlet; inletValue uniform {concentration:.17g}; value uniform {concentration:.17g}; }}
 ymin {{ type zeroGradient; }}
 ymax {{ type zeroGradient; }}
 zmin {{ type zeroGradient; }}
 zmax {{ type zeroGradient; }}
}}
''')
    patch_blockmesh(cfd/'system/blockMeshDict')

def main():
    global REPO, DATA, EXPOSURE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-repo', type=Path, default=REPO, help='PrelimDFG_GNN checkout (default: sibling checkout)')
    args=ap.parse_args()
    REPO=args.source_repo.resolve()
    DATA=REPO/'data/generation/study180_v1'
    EXPOSURE=REPO/'results/study180/exposure_summary_post.json'
    if not (DATA/'manifest.csv').is_file() or not EXPOSURE.is_file():
        raise FileNotFoundError(f'Expected study180_v1 data and exposure results under {REPO}')
    manifest = list(csv.DictReader((DATA/'manifest.csv').open()))
    exposure = json.loads(EXPOSURE.read_text())
    pool=[]
    for row in manifest:
        if int(row['n']) != 50: continue
        case=row['case']; value=exposure[case][-1]['mean_geometric_exposure']
        pool.append((value,case,row))
    pool.sort()
    # Twelve evenly spaced ranks, three exposure strata; all have 50 primary spheres.
    chosen=[pool[i] for i in (0,4,8,12,16,20,24,28,32,36,40,44)]
    lo,hi=pool[0][0],pool[-1][0]
    cuts=((pool[14][0]+pool[15][0])/2,(pool[29][0]+pool[30][0])/2)
    # Cases are assigned exposure groups by thirds of the full N=50 distribution.
    base=json.loads((HERE/'settings.json').read_text())
    base['uptake_duration_s']=1.6e-4
    base['padding_diameters']=4.0
    base['geometry']=''
    geom_dir=HERE/'geometries'; geom_dir.mkdir(exist_ok=True)
    case_root=HERE/'cases'
    matrix=[]
    for value, name, row in chosen:
        source=DATA/name/'particles.csv'
        local=geom_dir/f'{name}.csv'
        shutil.copyfile(source,local)
        prov={
          'source_repository':'PrelimDFG_GNN', 'source_commit':__import__('subprocess').check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
          'source_file':f'data/generation/study180_v1/{name}/particles.csv','source_sha256':sha(source),
          'particle_count':int(row['n']),'family':row['family'],'structural_class':row['structural_class'],
          'mean_geometric_exposure':value,'exposure_rays':512,'exposure_step':20000,
          'exposure_source_dump_sha256':exposure[name][-1]['source_dump_sha256'],
          'dem_relaxation':'20,000 BPM steps; source geometries retained in study180_v1; see results/study180/dem_summary.csv',
          'warning':'Exposure is a geometry-only descriptor, not a transport prediction.'}
        local.with_suffix('.provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
        group='low' if value < cuts[0] else ('middle' if value < cuts[1] else 'high')
        # use aggregate bounding-box diameter along streamwise direction for Pe.
        pts=list(csv.DictReader(local.open()))
        xs=[float(p['x_m']) for p in pts]; rs=[float(p['radius_m']) for p in pts]
        L=max(xs[i]+rs[i] for i in range(len(xs)))-min(xs[i]-rs[i] for i in range(len(xs)))
        for pe in (0,1,10,50):
            settings=dict(base)
            D=settings['diffusivity_m2_s']; U=pe*D/L if pe else 0.0
            relative_geometry=f'../../../geometries/{name}.csv'
            settings.update({'geometry':relative_geometry,'peclet_number':pe,'characteristic_length_m':L,'inlet_velocity_m_s':U})
            rel=Path(name)/f'Pe{pe:02d}'
            out=case_root/rel
            run=prepare(settings,local,out)
            run['source_geometry']=relative_geometry
            (out/'case-settings.json').write_text(json.dumps(settings,indent=2)+'\n')
            set_flow_bcs(out,U,settings['initial_concentration_mol_m3'])
            run.update({'flow':'steady laminar x-directed inlet/outlet; fixed agglomerate; no DEM integration',
                        'peclet_definition':'U_in * streamwise aggregate bounding-box length / molecular diffusivity',
                        'peclet_number':pe,'inlet_velocity_m_s':U,'characteristic_length_m':L,
                        'reynolds_number':U*L/1e-6,'exposure_group':group,'accuracy_certified':False,
                        'status':'prepared_unrun','prepared_case_directory':str(rel)})
            (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            matrix.append({'geometry_id':name,'exposure_group':group,'mean_geometric_exposure':value,
                           'family':row['family'],'N':50,'Pe':pe,'L_m':L,'U_m_s':U,
                           'Re':U*L/1e-6,'case_directory':str(rel),'status':'prepared_unrun'})
    with (case_root/'matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=matrix[0].keys());w.writeheader();w.writerows(matrix)
    summary={'selected_geometries':len(chosen),'prepared_cases':len(matrix),'Pe_values':[0,1,10,50],
             'exposure_group_cutoffs':list(cuts),'run_launched':False,
             'selection':'12 evenly spaced exposure ranks among N=50 geometries in the 180-case DEM/GNN study set'}
    (case_root/'matrix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    for x in chosen: print(f'{x[1]} exposure={x[0]:.6f} family={x[2]["family"]}')

if __name__=='__main__': main()
