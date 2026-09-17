#!/usr/bin/env python3
"""Convert native LAMMPS dumps, without third-party packages or unit conversion.

Run in DEM/post: python3 /path/to/posttovtk.py [options] [post_*.txt]
Defaults: legacy VTK series for particles and domain; PVD is optional.
Edit DEFAULT_PARTICLES/DEFAULT_DOMAIN below or override with --particles/--domain
(vtk, pvd, both, none). PVD references XML files kept in vtk_data/.
Domain is the LAMMPS bounding box, not a cylinder/STL wall or CFD mesh.
Physical ITEM: TIME takes priority; otherwise --dt maps step to time, or the
series uses timestep numbers with a warning. --dt assumes time zero at step 0.
"""
import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# User-editable output defaults; CLI options override these.
DEFAULT_PARTICLES = "vtk"
DEFAULT_DOMAIN = "vtk"


def frames(path):
    with path.open() as stream:
        step = time = count = bounds = None
        for line in stream:
            if line.startswith('ITEM: TIME\n'):
                time = float(next(stream))
            elif line.startswith('ITEM: TIMESTEP'):
                step = int(next(stream))
            elif line.startswith('ITEM: NUMBER OF ATOMS'):
                count = int(next(stream))
            elif line.startswith('ITEM: BOX BOUNDS'):
                if any(key in line.split()[3:] for key in ('xy', 'xz', 'yz', 'abc', 'origin')):
                    raise ValueError(f'{path}: only orthogonal BOX BOUNDS are supported')
                bounds = [tuple(map(float, next(stream).split())) for _ in range(3)]
                if any(len(b) != 2 or not all(map(math.isfinite, b)) or b[0] >= b[1] for b in bounds):
                    raise ValueError(f'{path}: invalid orthogonal box bounds')
            elif line.startswith('ITEM: ATOMS '):
                columns = line.split()[2:]
                if step is None or count is None or count < 0 or bounds is None:
                    raise ValueError(f'{path}: incomplete dump header')
                rows = [next(stream).split() for _ in range(count)]
                if any(len(row) != len(columns) for row in rows):
                    raise ValueError(f'{path}: incomplete atom record')
                yield step, time, columns, rows, bounds
                step = time = count = bounds = None


def convert(path):
    snapshots = list(frames(path))
    if not snapshots:
        raise ValueError(f'{path}: no custom atom snapshots')
    for step, time, columns, rows, bounds in snapshots:
        if not all(c in columns for c in ('x', 'y', 'z')):
            raise ValueError(f'{path}: requires physical x/y/z columns (not xs/ys/zs)')
        if any(not math.isfinite(float(v)) for row in rows for v in row):
            raise ValueError(f'{path}: non-finite or non-numeric atom data')
        output = path.with_suffix('.vtk') if len(snapshots) == 1 else path.with_name(f'{path.stem}_{step}.vtk')
        n = len(rows)
        with output.open('w') as out:
            out.write('# vtk DataFile Version 3.0\nLAMMPS particles; original dump units\nASCII\nDATASET POLYDATA\n')
            if time is not None:
                out.write(f'FIELD FieldData 1\nTIME 1 1 double\n{time:.17g}\n')
            out.write(f'POINTS {n} double\n')
            xyz = [columns.index(c) for c in ('x', 'y', 'z')]
            for row in rows:
                out.write(' '.join(row[i] for i in xyz) + '\n')
            out.write(f'VERTICES {n} {2*n}\n')
            for i in range(n):
                out.write(f'1 {i}\n')
            out.write(f'POINT_DATA {n}\n')
            for i, name in enumerate(columns):
                # Integer IDs are written from their original tokens, never rounded
                # through a double. Preserve every numeric column as a scalar.
                vtk_name = re.sub(r'[^A-Za-z0-9_]', '_', name)
                integer = name in ('id', 'type')
                out.write(f'SCALARS {vtk_name} {"long" if integer else "double"} 1\nLOOKUP_TABLE default\n')
                for row in rows:
                    out.write((str(int(row[i])) if integer else row[i]) + '\n')
            vectors = {
                'velocity': ('vx', 'vy', 'vz'),
                'angularVelocity': ('omegax', 'omegay', 'omegaz'),
                'totalForce': ('fx', 'fy', 'fz'),
                'hydrodynamicForce': tuple(f'c_hydroForce[{i}]' for i in (1, 2, 3)),
                'hydrodynamicTorque': tuple(f'c_hydroTorque[{i}]' for i in (1, 2, 3)),
            }
            for name, components in vectors.items():
                if not all(c in columns for c in components):
                    continue
                indices = [columns.index(c) for c in components]
                out.write(f'VECTORS {name} double\n')
                for row in rows:
                    out.write(' '.join(row[i] for i in indices) + '\n')
        print(output)


def array(parent, name, values, components=1, dtype='Float64'):
    element = ET.SubElement(parent, 'DataArray', type=dtype, Name=name,
                            NumberOfComponents=str(components), format='ascii')
    element.text = ' '.join(str(v) for v in values)


def polydata(points, time, nverts=0, nlines=0):
    root = ET.Element('VTKFile', type='PolyData', version='0.1', byte_order='LittleEndian')
    data = ET.SubElement(root, 'PolyData')
    if time is not None:
        if not math.isfinite(time):
            raise ValueError('Non-finite snapshot time')
        field = ET.SubElement(data, 'FieldData')
        array(field, 'TimeValue', [format(time, '.17g')])
        field[0].set('NumberOfTuples', '1')
    piece = ET.SubElement(data, 'Piece', NumberOfPoints=str(len(points)),
                          NumberOfVerts=str(nverts), NumberOfLines=str(nlines),
                          NumberOfStrips='0', NumberOfPolys='0')
    array(ET.SubElement(piece, 'Points'), 'Points',
          (v for point in points for v in point), 3)
    return root, piece


def write_xml(path, columns, rows, time):
    n = len(rows)
    xyz = [columns.index(c) for c in ('x', 'y', 'z')]
    root, piece = polydata([[row[i] for i in xyz] for row in rows], time, nverts=n)
    verts = ET.SubElement(piece, 'Verts')
    array(verts, 'connectivity', range(n), dtype='Int64')
    array(verts, 'offsets', range(1, n+1), dtype='Int64')
    data = ET.SubElement(piece, 'PointData')
    for i, name in enumerate(columns):
        array(data, re.sub(r'[^A-Za-z0-9_]', '_', name),
              (row[i] for row in rows), dtype='Int64' if name in ('id', 'type') else 'Float64')
    vectors = {
        'velocity': ('vx', 'vy', 'vz'),
        'angularVelocity': ('omegax', 'omegay', 'omegaz'),
        'totalForce': ('fx', 'fy', 'fz'),
        'hydrodynamicForce': tuple(f'c_hydroForce[{i}]' for i in (1, 2, 3)),
        'hydrodynamicTorque': tuple(f'c_hydroTorque[{i}]' for i in (1, 2, 3)),
    }
    for name, components in vectors.items():
        if all(c in columns for c in components):
            indices = [columns.index(c) for c in components]
            array(data, name, (row[i] for row in rows for i in indices), 3)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def box_geometry(bounds):
    points = [(bounds[0][i & 1], bounds[1][(i >> 1) & 1], bounds[2][(i >> 2) & 1])
              for i in range(8)]
    edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if not i & bit]
    return points, edges


def write_box(path, bounds, time):
    points, edges = box_geometry(bounds)
    root, piece = polydata(points, time, nlines=12)
    lines = ET.SubElement(piece, 'Lines')
    array(lines, 'connectivity', (v for edge in edges for v in edge), dtype='Int64')
    array(lines, 'offsets', range(2, 25, 2), dtype='Int64')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def write_box_legacy(path, bounds):
    points, edges = box_geometry(bounds)
    with path.open('w') as out:
        out.write('# vtk DataFile Version 3.0\nLAMMPS box borders; original dump units\nASCII\nDATASET POLYDATA\nPOINTS 8 double\n')
        for point in points:
            out.write(' '.join(format(v, '.17g') for v in point) + '\n')
        out.write('LINES 12 36\n')
        for i, j in edges:
            out.write(f'2 {i} {j}\n')


def collection(path, entries):
    root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    group = ET.SubElement(root, 'Collection')
    for time, name in sorted(entries):
        ET.SubElement(group, 'DataSet', timestep=format(time, '.17g'),
                      group='', part='0', file=name)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ('vtk', 'pvd', 'both', 'none')
    parser.add_argument('--particles', choices=choices, default=DEFAULT_PARTICLES)
    parser.add_argument('--domain', choices=choices, default=DEFAULT_DOMAIN)
    parser.add_argument('--dt', type=float, help='Dump-unit timestep for dumps without ITEM: TIME')
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()
    if args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0):
        parser.error('--dt must be finite and positive')
    files = args.files or sorted(Path.cwd().glob('post_*.txt'))
    if not files:
        parser.error('No post_*.txt files; run inside DEM/post or supply filenames.')
    try:
        groups = {}
        warned = False
        for path in sorted(set(p.resolve() for p in files)):
            snapshots = list(frames(path))
            if not snapshots:
                raise ValueError(f'{path}: no custom atom snapshots')
            for step, time, columns, rows, bounds in snapshots:
                if not all(c in columns for c in ('x', 'y', 'z')):
                    raise ValueError(f'{path}: requires physical x/y/z columns')
                if any(not math.isfinite(float(v)) for row in rows for v in row):
                    raise ValueError(f'{path}: non-finite or non-numeric atom data')
                if time is None:
                    time = step * args.dt if args.dt is not None else step
                    if args.dt is None and not warned:
                        print('Warning: missing ITEM: TIME; using timestep numbers. '
                              'Supply --dt for physical times.', file=sys.stderr)
                        warned = True
                if not math.isfinite(time):
                    raise ValueError(f'{path}: invalid snapshot time')
                stem = path.stem if len(snapshots) == 1 else f'{path.stem}_{step}'
                groups.setdefault(path.parent, []).append((time, stem, columns, rows, bounds))
        # Validate series before creating any outputs.
        for folder, records in groups.items():
            if len({r[0] for r in records}) != len(records):
                raise ValueError(f'{folder}: duplicate snapshot times; select one dump series')
            if len({r[1] for r in records}) != len(records):
                raise ValueError(f'{folder}: duplicate output names')
        if args.particles in ('vtk', 'both'):
            for path in sorted(set(p.resolve() for p in files)):
                convert(path)
        for folder, records in groups.items():
            particle_entries, box_entries = [], []
            for time, stem, columns, rows, bounds in sorted(records):
                if args.particles in ('pvd', 'both'):
                    target = folder/'vtk_data'/f'{stem}.vtp'
                    target.parent.mkdir(exist_ok=True)
                    write_xml(target, columns, rows, time)
                    particle_entries.append((time, target.relative_to(folder).as_posix()))
                if args.domain in ('vtk', 'both'):
                    target = folder/f'box_{stem}.vtk'
                    write_box_legacy(target, bounds)
                    print(target)
                if args.domain in ('pvd', 'both'):
                    target = folder/'vtk_data'/f'box_{stem}.vtp'
                    target.parent.mkdir(exist_ok=True)
                    write_box(target, bounds, time)
                    box_entries.append((time, target.relative_to(folder).as_posix()))
            if box_entries:
                collection(folder/'box.pvd', box_entries)
            if particle_entries:
                collection(folder/'particles.pvd', particle_entries)
    except (ValueError, OSError, StopIteration) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()

