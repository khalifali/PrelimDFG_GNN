#!/usr/bin/env python3
"""Generate connected monodisperse aggregates; all internal lengths are in radii."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

VERSION = '1.0'
TOL = 1e-9  # Relative to the primary-particle radius, not an SI length.


def direction(rng):
    """An isotropic direction, obtained by normalising a Gaussian vector."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def rotation(rng):
    """Uniform proper rotation from a normalised Gaussian quaternion."""
    w, x, y, z = v = rng.normal(size=4)
    w, x, y, z = v / np.linalg.norm(v)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def first_contact(a, b, axis, offset):
    """Slide b + offset + t*axis from t=+infinity towards fixed a.

    Every pair defines an interval of t in which its spheres overlap.
    The largest upper endpoint is the first collision of the whole clusters.
    No time stepping is used, so a particle cannot tunnel through a neighbour.
    Return None when the complete trajectory misses the fixed cluster.
    """
    delta = a[:, None, :] - b[None, :, :] - offset
    along = delta @ axis
    perpendicular = delta - along[..., None] * axis
    discriminant = 4.0 - np.sum(perpendicular**2, axis=-1)
    hit = discriminant >= 0
    if not hit.any():
        return None
    t = np.max(along[hit] + np.sqrt(discriminant[hit]))
    return b + offset + t * axis


def collide(a, b, rng, max_trials=10000):
    """Random orientations and area-uniform impact offsets; reject misses.

    Orientation is held fixed during one approach. The impact disk encloses
    both projected clusters. Directions and offsets are redrawn after misses;
    accepted collisions are consequently weighted by projected collision area.
    """
    a = (a - a.mean(axis=0)) @ rotation(rng).T
    b = (b - b.mean(axis=0)) @ rotation(rng).T
    bound = np.linalg.norm(a, axis=1).max() + np.linalg.norm(b, axis=1).max() + 2
    for _ in range(max_trials):
        axis = direction(rng)
        helper = np.eye(3)[np.argmin(abs(axis))]
        e1 = np.cross(axis, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        angle = rng.uniform(0, 2*np.pi)
        offset = bound * np.sqrt(rng.random()) * (np.cos(angle)*e1 + np.sin(angle)*e2)
        moved = first_contact(a, b, axis, offset)
        if moved is not None:
            joined = np.vstack([a, moved])
            return joined - joined.mean(axis=0)
    raise RuntimeError('Collision search exhausted; no substitute geometry was generated.')


def bpca(n, rng):
    """Ballistic particle-cluster growth: one incoming monomer per collision."""
    points = np.zeros((1, 3))
    for _ in range(n-1):
        points = collide(points, np.zeros((1, 3)), rng)
    return points


def bcca(n, rng):
    """Balanced ballistic cluster-cluster growth, allowing arbitrary N.

    Split N recursively into floor(N/2) and ceil(N/2). This is equal-mass
    hierarchical BCCA for powers of two, and a near-equal-mass variant otherwise.
    It is not the prescribed-Df cluster generator used in the original draft.
    """
    if n == 1:
        return np.zeros((1, 3))
    return collide(bcca(n//2, rng), bcca(n-n//2, rng), rng)


def local_filling(n, rng, detector_radius=6.0, max_trials=5000):
    """Attach to a least-crowded particle, following the Ringl-Urbassek idea.

    Equal spheres make neighbour counts proportional to local filling factor.
    Counts exclude the central particle. Only minimum-count sites are tried;
    failure is explicit, rather than silently selecting a denser growth site.
    """
    points = np.zeros((n, 3))
    for k in range(1, n):
        current = points[:k]
        d2 = np.sum((current[:, None] - current[None, :])**2, axis=-1)
        counts = np.sum(d2 <= detector_radius**2, axis=1) - 1
        candidates = rng.permutation(np.flatnonzero(counts == counts.min()))
        accepted = False
        for idx in candidates:
            for _ in range(max_trials):
                p = current[idx] + 2*direction(rng)
                if np.all(np.linalg.norm(current-p, axis=1) >= 2-TOL):
                    points[k] = p
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            raise RuntimeError(f'Local filling search exhausted at N={k}.')
    return points - points.mean(axis=0)


def inspect(points):
    """Independent all-pairs overlap and connected-component checks."""
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Coordinates must be a finite N by 3 array.')
    n = len(points)
    dist = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
    i, j = np.triu_indices(n, 1)
    gaps = dist[i, j]-2
    if np.any(gaps < -TOL):
        raise ValueError(f'Overlap exceeds tolerance: {gaps.min()} radii')
    edges = np.column_stack([i[abs(gaps) <= TOL], j[abs(gaps) <= TOL]])
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = {0}
    todo = [0]
    while todo:
        for v in adjacency[todo.pop()]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    if len(seen) != n:
        raise ValueError('Aggregate is disconnected.')
    centered = points - points.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T @ centered / n)
    # Rg includes the 3*r^2/5 contribution from the volume of each solid sphere.
    rg = np.sqrt(eig.sum() + 0.6)
    anisotropy = 0.0 if eig.sum() == 0 else float(1.5*np.sum(eig**2)/eig.sum()**2 - 0.5)
    metrics = dict(n=n, contacts=len(edges), connected=True,
                   max_overlap_over_radius=float(max(0, -gaps.min())) if len(gaps) else 0.0,
                   rg_over_radius=float(rg), anisotropy=anisotropy,
                   mean_coordination=2*len(edges)/n)
    return metrics, edges


def write_vtk(path, points, radius):
    """ParaView: apply Glyph/Sphere, scale by radius and use scale factor 2."""
    with path.open('w') as f:
        f.write('# vtk DataFile Version 3.0\nAggregate, SI metres\nASCII\nDATASET POLYDATA\n')
        f.write(f'POINTS {len(points)} double\n')
        np.savetxt(f, points, fmt='%.17g')
        f.write(f'VERTICES {len(points)} {2*len(points)}\n')
        for i in range(len(points)):
            f.write(f'1 {i}\n')
        f.write(f'POINT_DATA {len(points)}\nSCALARS radius double 1\nLOOKUP_TABLE default\n')
        np.savetxt(f, np.full(len(points), radius), fmt='%.17g')


def campaign(args):
    if any(n < 2 for n in args.counts) or args.replicates < 1 or args.seed < 0:
        raise ValueError('Counts must be >=2, replicates >=1 and seed >=0.')
    if not np.isfinite(args.diameter_um) or args.diameter_um <= 0:
        raise ValueError('Diameter must be finite and positive.')
    if any(not np.isfinite(x) or x <= 2 for x in args.detectors):
        raise ValueError('Detector radii must be finite and greater than 2 particle radii.')
    if len(set(args.counts)) != len(args.counts) or len(set(args.detectors)) != len(args.detectors):
        raise ValueError('Duplicate parameter values are not allowed.')
    # Never mix a partially generated campaign with an older one.
    args.output.mkdir(parents=True, exist_ok=False)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    families = [('bpca', None), ('bcca', None)] + [('local_filling', d) for d in args.detectors]
    manifest = []
    for method, detector in families:
        family = method if detector is None else f'{method}_r{detector:g}'
        for n in args.counts:
            for rep in range(args.replicates):
                case = f'{family}_N{n:04d}_rep{rep+1:02d}'
                # Stable case seeds do not depend on campaign ordering or size.
                seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{case}'.encode()).digest()[:8], 'little')
                rng = np.random.default_rng(seed)
                points = local_filling(n, rng, detector) if detector is not None else globals()[method](n, rng)
                metrics, edges = inspect(points)
                radius = args.diameter_um*1e-6/2
                folder = args.output / case
                folder.mkdir()
                physical = points*radius
                np.savetxt(folder/'particles.csv', np.column_stack([physical, np.full(n, radius)]),
                           delimiter=',', header='x_m,y_m,z_m,radius_m', comments='', fmt='%.17g')
                np.savetxt(folder/'contacts.csv', edges+1, delimiter=',', header='id_i,id_j', comments='', fmt='%d')
                write_vtk(folder/'particles.vtk', physical, radius)
                row = dict(case=case, method=method, family=family, detector_radius_over_r=detector,
                           replicate=rep+1, seed=seed, diameter_um=args.diameter_um, **metrics)
                (folder/'metadata.json').write_text(json.dumps(row, indent=2)+'\n')
                manifest.append(row)
    with (args.output/'manifest.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = dict(generator_version=VERSION, generator_sha256=source_sha,
                   numpy=np.__version__, python=platform.python_version(),
                   master_seed=args.seed, case_count=len(manifest), all_passed=True,
                   counts=args.counts, replicates=args.replicates,
                   diameter_um=args.diameter_um, detector_radii_over_r=args.detectors,
                   overlap_tolerance_over_radius=TOL)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[50, 100, 200])
    parser.add_argument('--replicates', type=int, default=5)
    parser.add_argument('--detectors', type=float, nargs='+', default=[3, 6, 12])
    parser.add_argument('--diameter-um', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=18427)
    campaign(parser.parse_args())
