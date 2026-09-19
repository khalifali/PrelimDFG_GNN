"""Target-blind coverage audit; bins describe occupancy, not feasibility or folds."""
import argparse,csv,hashlib,json
from pathlib import Path

def audit(source,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(source.open()))
    assert len(rows)==359 and len({r['case'] for r in rows})==359
    def write(name,data):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary=[]
    for n in sorted({int(r['n']) for r in rows}):
        rr=[r for r in rows if int(r['n'])==n]
        s=dict(n=n,cases=len(rr),groups=len({r['group'] for r in rr}),quality_ok=sum(r['box_quality']=='ok' for r in rr))
        for key in ('box_dimension','hull_porosity','asphericity','branch_fraction','local_density_cv'):
            s[key+'_min']=min(float(r[key]) for r in rr);s[key+'_max']=max(float(r[key]) for r in rr)
        summary.append(s)
    write('coverage_by_size.csv',summary)
    # Fixed descriptive bins. Empty cells must not be interpreted as attainable.
    db=[1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1]
    pb=[.65,.70,.75,.80,.85,.90,.95,1.000001]
    cells=[]
    for n in sorted({int(r['n']) for r in rows}):
        for dl,dh in zip(db,db[1:]):
            for pl,ph in zip(pb,pb[1:]):
                rr=[r for r in rows if int(r['n'])==n and dl<=float(r['box_dimension'])<dh and pl<=float(r['hull_porosity'])<ph]
                cells.append(dict(n=n,Dbox_low=dl,Dbox_high=dh,porosity_low=pl,porosity_high=ph,cases=len(rr),quality_ok=sum(r['box_quality']=='ok' for r in rr)))
    assert sum(c['cases'] for c in cells)==359
    write('joint_coverage_cells.csv',cells)
    lines=['# Existing BPM database: measured coverage audit','',
        'Source: committed final-geometry structural_features.csv from the 359-case development dataset. No exposure labels enter this audit. All cases are previously inspected development data.','',
        '| N | Cases | Box-quality flags | Measured Dbox range | Hull porosity range |','|---:|---:|---:|---|---|']
    for s in summary:
        lines.append(f"| {s['n']} | {s['cases']} | {s['cases']-s['quality_ok']} | {s['box_dimension_min']:.4f}–{s['box_dimension_max']:.4f} | {s['hull_porosity_min']:.4f}–{s['hull_porosity_max']:.4f} |")
    lines+=['','## Findings and generation priorities','',
        '98 of 359 estimates have box-quality flags. 89 flags occur at N=50,75,100. Do not treat apparent low-N dimension differences as well-resolved fractal regimes. Retain these geometries but carry their flags into coverage and reporting.',
        '',
        'Measured Dbox spans about 1.74–2.31, not the requested-generation exponent range. No current case reaches Dbox 2.5 or higher or 1.5. These gaps do not prove a generator cannot make compact/open structures: finite-size and estimator effects must be separated from geometry.',
        '',
        'Coverage depends strongly on size: for N>=350 all existing porosities exceed 0.84, while N=50 includes porosities near 0.73. Prioritize compact, lower-porosity candidates at medium/large N; more open candidates at small/medium N; and additional dimension variation within size classes where the estimator is reliable.',
        '',
        'joint_coverage_cells.csv records both empty and occupied fixed descriptor bins by N, including unflagged counts. These bins are descriptive, not split groups, physical feasibility claims, or a demand for a rectangular grid. coverage_by_size.csv additionally records anisotropy, branching and local-density ranges.',
        '',
        'Next gate: a small generator-feasibility pilot using the current generator, with reference geometries to test the box estimator. Classify final candidates using BPM-relaxed measured geometry. Increase independent seeds only after candidate configurations demonstrate new reliable coverage. FracVAL stays paused.',
        '',
        'Do not set a final database size yet. Freeze accepted coverage and the validation/test construction protocol before generating reserved fresh tests. No matched-pair splits will be used in the restored study.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=len(rows),box_quality_flags=sum(r['box_quality']!='ok' for r in rows),bins=dict(Dbox=db,porosity=pb),status='development coverage, not independent validation'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.source,a.output)
