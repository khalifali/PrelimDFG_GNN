#!/usr/bin/env python3
"""Prepare, then optionally run, fixed-geometry oxygen uptake cases.

Usage:
    # Prepare one case without launching a solver
    python3 run.py --geometry geometries/structure.csv --output cases/my_case --prepare-only

    # Execute one already prepared case (after activating LAMFOAM_PLUGIN)
    python3 run.py --run-prepared cases/my_case

The matrix builder creates and configures the 48 project cases. This script
refuses to overwrite prepared directories; --run-prepared runs in place.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from analyse import analyse

HERE = Path(__file__).resolve().parent


def header(name):
    return f'FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n'


def replace(path, key, value):
    text, count = re.subn(r'\b'+key+r'\s+[^;]+;', key+' '+str(value)+';', path.read_text())
    if count != 1:
        raise ValueError(f'Expected one {key} entry in {path}')
    path.write_text(text)


def prepare(settings, geometry, output):
    # No hidden diameter rescaling: archived CSV length columns are SI metres.
    with geometry.open(newline='') as f:
        source = list(csv.DictReader(f))
    particles = []
    for i, row in enumerate(source, 1):
        xyzr = [float(row[k]) for k in ('x_m', 'y_m', 'z_m', 'radius_m')]
        ident = int(row.get('id', i))
        if ident <= 0 or not all(math.isfinite(v) for v in xyzr) or xyzr[3] <= 0:
            raise ValueError('Particle IDs, coordinates and radii must be valid and finite')
        particles.append((ident, *xyzr))
    if not particles or len({p[0] for p in particles}) != len(particles):
        raise ValueError('Require nonempty geometry with unique positive IDs')
    for key, value in settings.items():
        if key == 'geometry':
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid numeric setting {key}')
    positive = ['diffusivity_m2_s', 'initial_concentration_mol_m3', 'particle_density_kg_m3',
                'base_cells_per_min_diameter', 'markers_per_particle', 'delta_t_s',
                'uptake_duration_s', 'refine_interval_steps', 'write_interval_steps', 'max_cells']
    if any(settings[k] <= 0 for k in positive):
        raise ValueError('Physical coefficients, timestep, resolution and counts must be positive')
    integers = ['base_cells_per_min_diameter', 'amr_levels', 'markers_per_particle',
                'refine_interval_steps', 'write_interval_steps', 'max_cells']
    if any(int(settings[k]) != settings[k] for k in integers):
        raise ValueError('Mesh, marker and interval counts must be integers')
    if settings['amr_levels'] < 1 or settings['markers_per_particle'] < 12:
        raise ValueError('This tutorial requires AMR and at least twelve markers per particle')
    if settings['markers_per_particle']*len(particles)>100000:
        raise ValueError('Experimental marker limit is 100000 total; choose fewer markers per particle explicitly')
    dt, warm, duration = (settings[k] for k in ('delta_t_s', 'warmup_s', 'uptake_duration_s'))
    for value in (warm, duration):
        if abs(value/dt-round(value/dt)) > 1e-8:
            raise ValueError('Warm-up and uptake durations must be multiples of delta_t_s')
    end = warm+duration
    if round(end/dt) % settings['write_interval_steps']:
        raise ValueError('Final time must be a field output time')
    if warm and round(warm/dt) < (settings['amr_levels']+1)*settings['refine_interval_steps']:
        raise ValueError('Allow at least levels+1 adaptation intervals during warm-up')
    diameter = 2*min(p[4] for p in particles)
    dmax = 2*max(p[4] for p in particles)
    h = diameter/settings['base_cells_per_min_diameter']
    padding = max(settings['padding_diameters']*dmax,
                  settings['padding_diffusion_lengths']*math.sqrt(settings['diffusivity_m2_s']*duration))
    if padding < 3*h:
        raise ValueError('Domain padding must cover at least the three-cell marker support')
    lo = [min(p[d+1]-p[4] for p in particles) for d in range(3)]
    hi = [max(p[d+1]+p[4] for p in particles) for d in range(3)]
    cells = [math.ceil((b-a+2*padding)/h) for a, b in zip(lo, hi)]
    lengths = [n*h for n in cells]
    shift = [L/2-(a+b)/2 for L, a, b in zip(lengths, lo, hi)]
    if math.prod(cells) >= settings['max_cells']:
        raise ValueError('Base mesh already exceeds max_cells; revise explicit settings')
    # Refuse overwrites so independent agglomerate results cannot be mixed.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(HERE/'template/CFD', output/'CFD')
    cfd, dem = output/'CFD', output/'DEM'
    dem.mkdir()
    shutil.copyfile(geometry, output/'source_particles.csv')
    provenance = geometry.with_suffix('.provenance.json')
    if provenance.exists():
        shutil.copyfile(provenance, output/'source_provenance.json')
    with (output/'particles.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'x_m', 'y_m', 'z_m', 'radius_m'])
        for ident, x, y, z, r in particles:
            w.writerow([ident, x+shift[0], y+shift[1], z+shift[2], r])
    manifest = dict(settings=settings, source_geometry=str(geometry), particle_count=len(particles),
                    source_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
                    translation_m=shift, geometry_scale=1., base_cells=cells,
                    box_lengths_m=lengths, base_cell_width_m=h,
                    finest_cell_width_m=h/2**settings['amr_levels'], warmup_end_s=warm,
                    end_time_s=end, accuracy_certified=False, units='SI', status='prepared')
    (output/'run.json').write_text(json.dumps(manifest, indent=2)+'\n')
    block = cfd/'system/blockMeshDict'
    replace(block, 'convertToMeters', 1)
    Lx, Ly, Lz = lengths
    vertices = [(0,0,0),(Lx,0,0),(Lx,Ly,0),(0,Ly,0),
                (0,0,Lz),(Lx,0,Lz),(Lx,Ly,Lz),(0,Ly,Lz)]
    replace(block, 'vertices', '('+' '.join('('+ ' '.join(format(v,'.17g') for v in p)+')' for p in vertices)+')')
    block.write_text(block.read_text().replace('(40 40 40)', '('+' '.join(map(str,cells))+')'))
    control = cfd/'system/controlDict'
    for k, v in [('endTime',format(end,'.17g')),('deltaT',format(dt,'.17g')),
                 ('writeInterval',settings['write_interval_steps']),('writePrecision',17),('timePrecision',12)]:
        replace(control,k,v)
    props = cfd/'system/couplingProperties'
    replace(props,'demSubsteps',0)
    replace(props,'flowWarmupEndTime',format(warm,'.17g'))
    props.write_text(props.read_text()+'\n// Fixed geometry, with flow/AMR-only startup.\nadaptiveMesh true;\n')
    schemes = cfd/'system/fvSchemes'
    schemes.write_text(schemes.read_text().replace('default backward;', 'default Euler;'))
    solution = cfd/'system/fvSolution'
    solution.write_text(re.sub(r'oxygen\s*\{[^{}]*\}', 'oxygen { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; }', solution.read_text()))
    # Empty forcing dictionary: no inherited stirring or hidden acceleration.
    (cfd/'constant/fvModels').write_text(header('fvModels'))
    (cfd/'constant/dynamicMeshDict').write_text(header('dynamicMeshDict')+f'''
 topoChanger {{
 type refiner; libs ("libfvMeshTopoChangers.so");
 refineInterval {settings['refine_interval_steps']};
 field interFace; lowerRefineLevel 1e-6; upperRefineLevel 1.000001;
 maxRefinement {settings['amr_levels']}; maxCells {settings['max_cells']};
 nBufferLayers 2; correctFluxes (); dumpLevel true;
 }}
''')
    (cfd/'0/oxygen').write_text('FoamFile { version 2.0; format ascii; class volScalarField; object oxygen; }\n'
        'dimensions [0 -3 0 0 1 0 0];\n'+f"internalField uniform {settings['initial_concentration_mol_m3']:.17g};\n"
        +f"boundaryField {{ \".*\" {{ type fixedValue; value uniform {settings['initial_concentration_mol_m3']:.17g}; }} }}\n")
    (cfd/'constant/particleTransportProperties').write_text(header('particleTransportProperties')+f'''
// Perfect absorbing surface: cumulative inventory, no finite saturation capacity.
enabled true;
species {{ oxygen {{
 inventory oxygenInventory; exchangeModel surfaceMarkerExperimental;
 diffusivity [0 2 -1 0 0 0 0] {settings['diffusivity_m2_s']:.17g};
 surfaceSamples {settings['markers_per_particle']};
 maxMarkers {settings['markers_per_particle']*len(particles)};
 markerSupportCells 3; markerMixedCellSupport true;
 periodicAxes (0 0 0); budgetMode open;
 particleOutputDirectory "../DEM/oxygen";
}} }}
''')
    # Write explicit IDs and original physical radii in a LAMMPS sphere data file.
    data = ['Relaxed agglomerate; SI; translated only', '', f'{len(particles)} atoms', '1 atom types', '']
    data += [f'0 {L:.17g} {axis}lo {axis}hi' for L,axis in zip(lengths,'xyz')]
    data += ['', 'Atoms # sphere', '']
    for ident,x,y,z,r in particles:
        data.append(' '.join(map(str,[ident,1,2*r,settings['particle_density_kg_m3'],x+shift[0],y+shift[1],z+shift[2]])))
    (dem/'particles.data').write_text('\n'.join(data)+'\n')
    (dem/'in.lammps').write_text(f'''# Fixed BPM-relaxed geometry. Do not integrate or re-relax this agglomerate.
units si
atom_style sphere
atom_modify map array
boundary f f f
read_data ../DEM/particles.data
velocity all set 0 0 0
set group all omega 0 0 0
pair_style zero {2*dmax:.17g}
pair_coeff * *
neighbor {0.2*diameter:.17g} bin
timestep {dt:.17g}
thermo 100
# No integration fix and no positive run command: LAMFOAM owns time advancement.
''')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--settings', type=Path, default=HERE/'settings.json')
    ap.add_argument('--geometry', type=Path, help='Final relaxed CSV with x_m,y_m,z_m,radius_m, optional id')
    ap.add_argument('--output', type=Path, default=HERE/'runs/example')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--run-prepared', type=Path, help='Execute an existing prepared case directory')
    args = ap.parse_args()
    if args.run_prepared:
        output = args.run_prepared.resolve()
        manifest_path = output/'run.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') not in ('prepared', 'prepared_unrun'):
            raise RuntimeError(f"Case status is {manifest.get('status')}; expected prepared")
        if not os.environ.get('LAMFOAM_PLUGIN') or not Path(os.environ['LAMFOAM_PLUGIN']).is_file():
            raise RuntimeError('Activate the existing LAMFOAM environment with LAMFOAM_PLUGIN before running')
        env = os.environ.copy(); env['LAMFOAM_INPUT'] = str(output/'DEM/in.lammps'); env['LAMFOAM_DEM_OUTPUT'] = str(output/'DEM')
        try:
            for command, log in [('blockMesh','log.blockMesh'),('lamfoamIBSolver','log.solver')]:
                with (output/'CFD'/log).open('w') as f:
                    subprocess.run([command],cwd=output/'CFD',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            result = analyse(output)
            manifest['status'] = 'completed'; manifest['analysis'] = result
        except Exception:
            manifest['status'] = 'failed'
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        print('Uptake curve:',output/'postProcessing/uptake_mean.csv')
        return
    settings = json.loads(args.settings.read_text())
    geometry = args.geometry.resolve() if args.geometry else (args.settings.resolve().parent/settings['geometry']).resolve()
    output = args.output.resolve()
    manifest = prepare(settings,geometry,output)
    print('Prepared',output, 'base mesh',manifest['base_cells'],flush=True)
    if args.prepare_only:
        return
    if not os.environ.get('LAMFOAM_PLUGIN') or not Path(os.environ['LAMFOAM_PLUGIN']).is_file():
        raise RuntimeError('Activate the existing LAMFOAM environment with LAMFOAM_PLUGIN before running')
    env = os.environ.copy(); env['LAMFOAM_INPUT'] = str(output/'DEM/in.lammps'); env['LAMFOAM_DEM_OUTPUT'] = str(output/'DEM')
    try:
        for command, log in [('blockMesh','log.blockMesh'),('lamfoamIBSolver','log.solver')]:
            with (output/'CFD'/log).open('w') as f:
                subprocess.run([command],cwd=output/'CFD',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        result = analyse(output)
        manifest['status'] = 'completed'; manifest['analysis'] = result
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        (output/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Uptake curve:',output/'postProcessing/uptake_mean.csv')


if __name__ == '__main__':
    main()
