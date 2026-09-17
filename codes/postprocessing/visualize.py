#!/usr/bin/env python3
"""Join exposure to particle geometry in compact VTP/PVD files; optional VTK."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from exposure import geometric_exposure
import posttovtk as converter


def process(folder,rays,output_format):
    paths=list(folder.glob('post_*.txt'))
    paths.sort(key=lambda p:int(p.stem.split('_')[-1]))
    if not paths:raise ValueError(f'No particle dumps in {folder}')
    entries=[];summary=[]
    for path in paths:
        snapshots=list(converter.frames(path))
        if len(snapshots)!=1:raise ValueError('One snapshot per file is required')
        step,time,columns,rows,bounds=snapshots[0]
        if time is None:raise ValueError('Physical ITEM: TIME required; do not infer from step number')
        data=np.asarray(rows,dtype=float)
        if not np.isfinite(data).all():raise ValueError('Non-finite dump data')
        xyz=data[:,[columns.index(k) for k in ['x','y','z']]]
        radii=data[:,columns.index('radius')]
        exposure=geometric_exposure(xyz,radii,rays)
        ids=[int(row[columns.index('id')]) for row in rows]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate particle IDs')
        # Keep original raw dumps, but only essential display arrays in VTP.
        selected=['id','type','x','y','z','radius','geometric_exposure']
        enriched=[[row[columns.index(k)] for k in selected[:-1]]+[format(a,'.17g')]
                  for row,a in zip(rows,exposure)]
        out=folder/'vtk_data';out.mkdir(exist_ok=True)
        target=out/f'post_{step}.vtp'
        converter.write_xml(target,selected,enriched,time)
        # Coordinates already occur in Points; omit redundant coordinate scalars.
        tree=ET.parse(target);point_data=tree.find('.//PointData')
        for array in list(point_data):
            if array.get('Name') in ('x','y','z'):point_data.remove(array)
        point_data.set('Scalars','geometric_exposure')
        tree.write(target,encoding='utf-8',xml_declaration=True)
        entries.append((time,target.relative_to(folder).as_posix()))
        if output_format in ['vtk','both']:
            # Reuse the established legacy writer without retaining a duplicate dump.
            with tempfile.TemporaryDirectory() as temp:
                tmp=Path(temp)/path.name
                with tmp.open('w') as f:
                    f.write(f'ITEM: TIME\n{time:.17g}\nITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\nITEM: BOX BOUNDS ff ff ff\n')
                    for pair in bounds:f.write(' '.join(map(str,pair))+'\n')
                    f.write('ITEM: ATOMS '+' '.join(selected)+'\n')
                    for row in enriched:f.write(' '.join(row)+'\n')
                converter.convert(tmp)
                (folder/f'post_{step}.vtk').write_bytes(tmp.with_suffix('.vtk').read_bytes())
        analysis=folder/'exposure';analysis.mkdir(exist_ok=True)
        with (analysis/f'exposure_{step}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['id','geometric_exposure']);w.writerows(zip(ids,exposure))
        summary.append(dict(step=step,time_us=time,n=len(rows),rays=rays,
                            mean_geometric_exposure=float(exposure.mean()),
                            source_dump_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    converter.collection(folder/'particles.pvd',entries)
    (folder/'exposure'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path,help='One post directory or a prepared DEM campaign')
    p.add_argument('--stage',choices=['post','generated'],default='post')
    p.add_argument('--rays',type=int,default=512)
    p.add_argument('--format',choices=['pvd','vtk','both'],default='pvd',help='VTK adds legacy files; VTP/PVD always retained')
    args=p.parse_args()
    if (args.path/'campaign.json').exists():
        campaign=json.loads((args.path/'campaign.json').read_text());all_summaries={}
        for case in campaign['cases']:
            all_summaries[case]=process(args.path/case/args.stage,args.rays,args.format)
        (args.path/f'exposure_summary_{args.stage}.json').write_text(json.dumps(all_summaries,indent=2)+'\n')
        print(f'Exposure and visualization complete for {len(all_summaries)} cases.')
    else:process(args.path,args.rays,args.format)
