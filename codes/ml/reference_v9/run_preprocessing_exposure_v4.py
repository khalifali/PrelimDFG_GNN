#!/usr/bin/env python3
"""Select relaxed DEM snapshots and build graph and geometric-exposure data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError


STEP_RE = re.compile(r"(\d+)(?=\.vtk$)")
CASE_RE = re.compile(
    r"^case_(?P<case_id>\d+)_dp(?P<particle_diameter>[0-9.]+)_"
    r"N(?P<n_requested>\d+)(?:_(?P<morphology>[^_]+))?_"
    r"Df(?P<df_requested>[0-9.]+)(?:_kf(?P<kf_requested>[0-9.]+))?_"
    r"rep(?P<realization>\d+)$"
)


def vtk_legacy(path: Path):
    """Read the ASCII legacy-VTK subset used by the supplied converters."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = {"points": np.empty((0, 3)), "lines": [], "point": {}, "cell": {}}
    i = 0
    location = None
    count = 0
    while i < len(lines):
        s = lines[i].strip()
        p = s.split()
        if not p:
            i += 1
            continue
        key = p[0].upper()
        if key == "POINTS":
            n = int(p[1]); vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(map(float, lines[i].split())); i += 1
            out["points"] = np.asarray(vals[:3*n], float).reshape(n, 3)
            continue
        if key == "LINES":
            n = int(p[1]); i += 1
            for _ in range(n):
                q = list(map(int, lines[i].split())); out["lines"].append(q[1:]); i += 1
            continue
        if key == "POINT_DATA":
            location, count = "point", int(p[1]); i += 1; continue
        if key == "CELL_DATA":
            location, count = "cell", int(p[1]); i += 1; continue
        if key == "SCALARS" and location:
            name, ncomp = p[1], int(p[3]) if len(p) > 3 else 1
            i += 1
            if i < len(lines) and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                i += 1
            vals = []
            while len(vals) < count * ncomp:
                vals.extend(map(float, lines[i].split())); i += 1
            a = np.asarray(vals[:count*ncomp])
            out[location][name] = a if ncomp == 1 else a.reshape(count, ncomp)
            continue
        if key == "VECTORS" and location:
            name = p[1]; vals = []; i += 1
            while len(vals) < count * 3:
                vals.extend(map(float, lines[i].split())); i += 1
            out[location][name] = np.asarray(vals[:count*3]).reshape(count, 3)
            continue
        i += 1
    return out


def timed_files(folder: Path, prefix: str):
    ans = {}
    for p in folder.glob(f"{prefix}*.vtk"):
        m = STEP_RE.search(p.name)
        if m:
            ans[int(m.group(1))] = p
    return ans


def particle_ids(v):
    ids = v["point"].get("id")
    if ids is None:
        ids = v["point"].get("particle_id")
    return np.arange(len(v["points"]), dtype=int) if ids is None else ids.astype(int)


def particle_radii(v):
    for k in ("radius", "r"):
        if k in v["point"]:
            return np.asarray(v["point"][k], float)
    raise ValueError("Particle VTK needs POINT_DATA scalar 'radius' (or 'r').")


def particle_cluster_ids(v):
    """Return optional per-particle cluster IDs and the matched field name."""
    aliases = {name.lower(): name for name in v["point"]}
    for candidate in (
        "c_cluscont", "clust_id", "cluster_id", "clustid", "clusterid"
    ):
        actual = aliases.get(candidate)
        if actual is not None:
            values = np.asarray(v["point"][actual])
            if values.ndim != 1 or len(values) != len(v["points"]):
                raise ValueError(
                    f"POINT_DATA scalar '{actual}' has an invalid shape: {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"POINT_DATA scalar '{actual}' contains non-finite values.")
            return values.astype(np.int64), actual
    return None, None


def contact_pairs(v):
    cd = v["cell"]
    alternatives = [("particle_id_i", "particle_id_j"), ("id_i", "id_j"), ("id1", "id2")]
    for a, b in alternatives:
        if a in cd and b in cd:
            return {(min(int(x), int(y)), max(int(x), int(y))) for x, y in zip(cd[a], cd[b])}
    raise ValueError("Contact VTK needs CELL_DATA particle_id_i and particle_id_j.")


def contact_pair_rows(v):
    """Map each unordered particle-ID pair to its row in CELL_DATA."""
    cd = v["cell"]
    alternatives = [("particle_id_i", "particle_id_j"), ("id_i", "id_j"), ("id1", "id2")]
    for a, b in alternatives:
        if a in cd and b in cd:
            mapping = {}
            for row, (x, y) in enumerate(zip(cd[a], cd[b])):
                mapping.setdefault((min(int(x), int(y)), max(int(x), int(y))), row)
            return mapping, {a, b}
    raise ValueError("Contact VTK needs CELL_DATA particle_id_i and particle_id_j.")


def structure_integrity(particle_path: Path, contact_path: Path):
    """Check cluster IDs and connectivity of the exact contact graph used later."""
    pv, cv = vtk_legacy(particle_path), vtk_legacy(contact_path)
    ids = particle_ids(pv)
    cluster_ids, cluster_field = particle_cluster_ids(pv)
    unique_cluster_ids = (
        sorted({int(value) for value in cluster_ids})
        if cluster_ids is not None else []
    )

    id_to_idx = {int(pid): i for i, pid in enumerate(ids)}
    pair_rows, _ = contact_pair_rows(cv)
    adjacency = [set() for _ in ids]
    for pid_a, pid_b in pair_rows:
        if pid_a in id_to_idx and pid_b in id_to_idx:
            i, j = id_to_idx[pid_a], id_to_idx[pid_b]
            adjacency[i].add(j)
            adjacency[j].add(i)

    component_sizes = []
    unseen = set(range(len(ids)))
    while unseen:
        start = unseen.pop()
        size = 0
        queue = [start]
        while queue:
            node = queue.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)

    component_sizes.sort(reverse=True)
    n_components = len(component_sizes)
    largest_fraction = component_sizes[0] / len(ids) if len(ids) else 0.0
    cluster_id_available = cluster_ids is not None
    cluster_id_consistent = not cluster_id_available or len(unique_cluster_ids) == 1
    single_agglomerate = cluster_id_consistent and n_components == 1
    return {
        "cluster_id_available": cluster_id_available,
        "cluster_id_field": cluster_field,
        "unique_cluster_ids": unique_cluster_ids,
        "n_clusters_from_cluster_id": (
            len(unique_cluster_ids) if cluster_id_available else None
        ),
        "n_contact_components": n_components,
        "contact_component_sizes": component_sizes,
        "largest_component_fraction": float(largest_fraction),
        "single_agglomerate": single_agglomerate,
    }


def transition_metrics(old_p, new_p, old_c, new_c):
    a, b = vtk_legacy(old_p), vtk_legacy(new_p)
    ida, idb = particle_ids(a), particle_ids(b)
    ma = {int(k): x for k, x in zip(ida, a["points"])}
    mb = {int(k): x for k, x in zip(idb, b["points"])}
    common = sorted(set(ma) & set(mb))
    if not common:
        raise ValueError("No common particle IDs between consecutive snapshots.")
    disp = np.asarray([mb[k] - ma[k] for k in common])
    dref = 2.0 * float(np.median(particle_radii(b)))
    rms = float(np.sqrt(np.mean(np.sum(disp**2, axis=1))) / dref)
    maximum = float(np.max(np.linalg.norm(disp, axis=1)) / dref)
    ca, cb = contact_pairs(vtk_legacy(old_c)), contact_pairs(vtk_legacy(new_c))
    union = ca | cb
    jchange = 0.0 if not union else 1.0 - len(ca & cb) / len(union)
    return {"rms_displacement_over_d": rms, "max_displacement_over_d": maximum,
            "contact_jaccard_change": float(jchange)}


def select_snapshot(post, cont, rms_tol, contact_tol, stable_intervals):
    steps = sorted(set(post) & set(cont))
    if len(steps) < stable_intervals + 1:
        raise ValueError(f"Need at least {stable_intervals + 1} common timesteps; found {steps}")
    history = []
    selected = None
    for a, b in zip(steps[:-1], steps[1:]):
        m = transition_metrics(post[a], post[b], cont[a], cont[b])
        m.update({"from_step": a, "to_step": b})
        history.append(m)
        recent = history[-stable_intervals:]
        if len(recent) == stable_intervals and all(
            x["rms_displacement_over_d"] <= rms_tol and
            x["contact_jaccard_change"] <= contact_tol for x in recent
        ):
            selected = b
            break
    return (selected if selected is not None else steps[-1]), selected is not None, history


def get_cell_array(v, names, n, default=0.0):
    for name in names:
        if name in v["cell"]:
            a = np.asarray(v["cell"][name], float)
            return a
    return np.full(n, default, float)


def fibonacci_directions(number):
    """Deterministic, approximately uniform unit vectors on a sphere."""
    indices = np.arange(number, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / number
    phi = np.pi * (3.0 - np.sqrt(5.0)) * indices
    radial = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    return np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))


def line_of_sight_exposure(xyz, radii, samples=512, probe_radius=0.0):
    """Estimate each particle's outward-normal visible surface fraction.

    A sampled surface direction is exposed when the ray starting just outside
    the particle, along its outward normal, reaches infinity without
    intersecting another particle. Other particles are inflated by
    ``probe_radius``. This is a geometric line-of-sight proxy, not a diffusion
    or oxygen-uptake calculation.
    """
    if samples < 32:
        raise ValueError("Exposure sampling requires at least 32 directions")
    if probe_radius < 0:
        raise ValueError("Exposure probe radius must be non-negative")
    directions = fibonacci_directions(samples)
    exposure = np.empty(len(xyz), dtype=float)
    scale = max(float(np.median(radii)), 1.0e-30)
    epsilon = 1.0e-9 * scale
    for i, (center, radius) in enumerate(zip(xyz, radii)):
        mask = np.arange(len(xyz)) != i
        other_centers = xyz[mask]
        other_radii = radii[mask] + probe_radius
        origins = center + (radius + probe_radius + epsilon) * directions
        # For every ray and potential blocking sphere, compute the forward
        # projection and squared perpendicular distance to the ray.
        offset = other_centers[None, :, :] - origins[:, None, :]
        projection = np.einsum("snk,sk->sn", offset, directions)
        perpendicular_squared = np.einsum("snk,snk->sn", offset, offset) - projection**2
        blocked = np.any(
            (projection > 0.0)
            & (perpendicular_squared <= other_radii[None, :]**2),
            axis=1,
        )
        exposure[i] = np.mean(~blocked)
    return exposure


def graph_and_targets(particle_path, contact_path, beta,
                      exposure_samples=512, exposure_probe_radius_over_dref=0.0):
    pv, cv = vtk_legacy(particle_path), vtk_legacy(contact_path)
    xyz, ids, radii = pv["points"], particle_ids(pv), particle_radii(pv)
    id_to_idx = {int(k): i for i, k in enumerate(ids)}
    pair_rows, endpoint_names = contact_pair_rows(cv)
    pairs_id = sorted(pair_rows)
    retained = [(a, b, pair_rows[(a, b)]) for a, b in pairs_id if a in id_to_idx and b in id_to_idx]
    pairs = [(id_to_idx[a], id_to_idx[b]) for a, b, _ in retained]
    adjacency = [set() for _ in ids]
    for i, j in pairs:
        adjacency[i].add(j); adjacency[j].add(i)
    center = xyz.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((xyz-center)**2, axis=1))))
    dref = 2.0 * float(np.median(radii))
    exposure_probe_radius = exposure_probe_radius_over_dref * dref
    geometric_exposure = line_of_sight_exposure(
        xyz, radii, samples=exposure_samples,
        probe_radius=exposure_probe_radius,
    )
    radial = np.linalg.norm(xyz-center, axis=1) / max(rg, 1e-30)
    degree = np.asarray([len(a) for a in adjacency], float)
    if len(ids) >= 4:
        try:
            surface = set(map(int, ConvexHull(xyz, qhull_options="QJ").vertices))
        except QhullError:
            surface = set(range(len(ids)))
    else:
        surface = set(range(len(ids)))
    depth_map = {}
    q = deque((s, 0) for s in surface)
    while q:
        u, d = q.popleft()
        if u in depth_map: continue
        depth_map[u] = d
        q.extend((v, d+1) for v in adjacency[u] if v not in depth_map)
    depth = np.asarray([depth_map.get(i, len(ids)) for i in range(len(ids))], float)
    node_x = np.column_stack((radii/dref, radial, degree))
    edge_index = np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    if pairs:
        edge_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)
    dist = np.asarray([np.linalg.norm(xyz[i]-xyz[j])/dref for i, j in pairs], float)
    overlap = np.asarray([(radii[i]+radii[j]-np.linalg.norm(xyz[i]-xyz[j]))/dref for i, j in pairs], float)
    edge_columns = [dist, overlap]
    edge_feature_names = ["distance_over_dref", "geometric_overlap_over_dref"]
    n_contact_cells = len(cv["lines"])
    for name in sorted(cv["cell"]):
        if name in endpoint_names:
            continue
        values = np.asarray(cv["cell"][name])
        if values.ndim == 0 or len(values) != n_contact_cells:
            continue
        selected = values[[row for _, _, row in retained]] if retained else values[:0]
        if selected.ndim == 1:
            edge_columns.append(selected.astype(float))
            edge_feature_names.append(f"contact_{name}")
        else:
            selected = selected.reshape(len(selected), -1)
            suffixes = ["x", "y", "z"] if selected.shape[1] == 3 else [str(i) for i in range(selected.shape[1])]
            for column, suffix in enumerate(suffixes):
                edge_columns.append(selected[:, column].astype(float))
                edge_feature_names.append(f"contact_{name}_{suffix}")
    edge_attr = np.column_stack(edge_columns) if pairs else np.empty((0, len(edge_feature_names)))
    edge_attr = np.concatenate((edge_attr, edge_attr), axis=0) if len(edge_attr) else edge_attr
    targets = {"mean_graph_depth": float(depth.mean()),
               "shielded_fraction_h_ge_2": float(np.mean(depth >= 2)),
               "accessibility_score": float(np.mean(np.exp(-beta*depth))),
               "mean_geometric_exposure": float(np.mean(geometric_exposure))}
    metrics = {"n_particles": len(ids), "n_contacts": len(pairs), "d_ref": dref,
               "radius_of_gyration": rg, "mean_coordination": float(degree.mean()),
               "surface_fraction": len(surface)/len(ids),
               "exposure_definition": "outward_normal_line_of_sight",
               "exposure_samples_per_particle": int(exposure_samples),
               "exposure_probe_radius_over_dref": float(exposure_probe_radius_over_dref),
               "minimum_particle_geometric_exposure": float(np.min(geometric_exposure)),
               "p10_particle_geometric_exposure": float(np.percentile(geometric_exposure, 10)),
               "maximum_particle_geometric_exposure": float(np.max(geometric_exposure)),
               "edge_feature_names": edge_feature_names, **targets}
    arrays = dict(node_features=node_x.astype(np.float32), edge_index=edge_index,
                  edge_features=edge_attr.astype(np.float32), positions=xyz.astype(np.float32),
                  particle_ids=ids.astype(np.int64), surface=np.asarray([i in surface for i in range(len(ids))]),
                  graph_depth=depth.astype(np.float32),
                  geometric_exposure=geometric_exposure.astype(np.float32),
                  node_feature_names=np.asarray(["radius_over_dref", "radial_distance_over_rg", "coordination"]),
                  edge_feature_names=np.asarray(edge_feature_names),
                  target_names=np.asarray(list(targets)),
                  targets=np.asarray(list(targets.values()), np.float32))
    return arrays, metrics


def write_accessibility_vtk(path, arrays):
    xyz = arrays["positions"]; n = len(xyz)
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\nDEM accessibility proxy\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for x, y, z in xyz: f.write(f"{x:.12e} {y:.12e} {z:.12e}\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n): f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        for name, vals, dtype in (("particle_id", arrays["particle_ids"], "int"),
                                  ("is_hull_surface", arrays["surface"].astype(int), "int"),
                                  ("graph_depth", arrays["graph_depth"], "float"),
                                  ("geometric_exposure", arrays["geometric_exposure"], "float")):
            f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n")
            for v in vals: f.write(f"{v}\n")


def write_node_csv(path, arrays):
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["particle_id", "x", "y", "z", "is_hull_surface", "graph_depth",
                                      "geometric_exposure",
                                      "radius_over_dref", "radial_distance_over_rg", "coordination"])
        for pid, pos, surf, depth, exposure, feat in zip(
                arrays["particle_ids"], arrays["positions"], arrays["surface"],
                arrays["graph_depth"], arrays["geometric_exposure"], arrays["node_features"]):
            w.writerow([int(pid), *map(float, pos), int(surf), float(depth),
                        float(exposure), *map(float, feat)])


def case_metadata(name: str):
    """Parse design parameters encoded in a generated case-directory name."""
    match = CASE_RE.fullmatch(name)
    if not match:
        return {}
    values = match.groupdict()
    metadata = {
        "case_id": int(values["case_id"]),
        "particle_diameter_requested": float(values["particle_diameter"]),
        "n_particles_requested": int(values["n_requested"]),
        "fractal_dimension_requested": float(values["df_requested"]),
        "realization": int(values["realization"]),
    }
    if values.get("morphology") is not None:
        metadata["morphology_requested"] = values["morphology"]
    if values.get("kf_requested") is not None:
        metadata["fractal_prefactor_requested"] = float(values["kf_requested"])
    return metadata


def first_existing(candidates):
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def discover_cases(root: Path):
    """Discover every case and resolve its actual post/contact VTK directories."""
    case_dirs = sorted(
        (p for p in root.glob("case_*") if p.is_dir()),
        key=lambda p: case_metadata(p.name).get("case_id", 10**9),
    )
    for case in case_dirs:
        results = case / "results"
        cont_dir = first_existing([
            results / "cont" / "cont_vtk",
            results / "cont" / "contact_vtk",
        ])
        post_dir = first_existing([
            results / "post" / "post_vtk",
        ])
        yield case, post_dir, cont_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", nargs="?", type=Path, default=Path("."),
        help="Directory containing case_* folders (default: current directory)",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--post-prefix", default="post_")
    ap.add_argument("--contact-prefix", default="contact_")
    ap.add_argument("--rms-tol", type=float, default=1e-4)
    ap.add_argument("--contact-tol", type=float, default=0.01)
    ap.add_argument("--stable-intervals", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument(
        "--exposure-samples", type=int, default=512,
        help="Fibonacci surface directions per particle (default: 512).",
    )
    ap.add_argument(
        "--exposure-probe-radius-over-dref", type=float, default=0.0,
        help="Optional blocking-sphere inflation radius divided by d_ref.",
    )
    ap.add_argument("--require-converged", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve(); outroot = (args.output or root/"gnn_dataset").resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = list(discover_cases(root))
    if not cases:
        raise SystemExit(f"No case_* directories were found directly under {root}")
    print(f"Discovered {len(cases)} case directories under {root}")
    for case, post_dir, cont_dir in cases:
        name = case.name; dest = outroot/name; dest.mkdir(exist_ok=True)
        status = {"case": name, **case_metadata(name),
                  "post_vtk_directory": str(post_dir),
                  "contact_vtk_directory": str(cont_dir)}
        try:
            if not post_dir.is_dir():
                raise FileNotFoundError(f"Missing particle VTK directory: {post_dir}")
            if not cont_dir.is_dir():
                raise FileNotFoundError(
                    "Missing contact VTK directory; checked "
                    f"{case/'results'/'cont'/'cont_vtk'} and "
                    f"{case/'results'/'cont'/'contact_vtk'}"
                )
            post = timed_files(post_dir, args.post_prefix); cont = timed_files(cont_dir, args.contact_prefix)
            if not post:
                raise FileNotFoundError(f"No {args.post_prefix}*.vtk files in {post_dir}")
            if not cont:
                raise FileNotFoundError(f"No {args.contact_prefix}*.vtk files in {cont_dir}")
            common_steps = sorted(set(post) & set(cont))
            if not common_steps:
                raise ValueError(
                    f"No matching post/contact timesteps. Post={sorted(post)}, contact={sorted(cont)}"
                )
            status.update(n_post_timesteps=len(post), n_contact_timesteps=len(cont),
                          n_common_timesteps=len(common_steps),
                          first_common_timestep=common_steps[0],
                          last_common_timestep=common_steps[-1])
            step, converged, hist = select_snapshot(post, cont, args.rms_tol, args.contact_tol, args.stable_intervals)
            status.update(selected_timestep=step, converged=converged, convergence_history=hist)
            if args.require_converged and not converged:
                raise ValueError("No timestep passed the requested relaxation criteria")
            pfinal, cfinal = dest/"post_final.vtk", dest/"contact_final.vtk"
            shutil.copy2(post[step], pfinal); shutil.copy2(cont[step], cfinal)
            integrity = structure_integrity(pfinal, cfinal)
            status.update(integrity)
            if not integrity["single_agglomerate"]:
                raise ValueError(
                    "Selected snapshot is not one connected agglomerate: "
                    f"cluster IDs={integrity['unique_cluster_ids'] if integrity['cluster_id_available'] else 'not available'}, "
                    f"contact-component sizes={integrity['contact_component_sizes']}"
                )
            arrays, metrics = graph_and_targets(
                pfinal, cfinal, args.beta,
                exposure_samples=args.exposure_samples,
                exposure_probe_radius_over_dref=args.exposure_probe_radius_over_dref,
            )
            np.savez_compressed(dest/"graph.npz", **arrays)
            write_accessibility_vtk(dest/"particles_accessibility.vtk", arrays)
            write_node_csv(dest/"node_metrics.csv", arrays)
            (dest/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            status.update(metrics); status["status"] = "ok"
        except Exception as exc:
            status.update(status="error", error=str(exc))
        (dest/"selection.json").write_text(json.dumps(status, indent=2)+"\n")
        rows.append(status)
        if status["status"] == "ok":
            print(
                f"{name}: ok, step={status.get('selected_timestep')}, "
                f"relaxed={status.get('converged')}, "
                f"single_agglomerate={status.get('single_agglomerate')}, "
                f"contact_components={status.get('n_contact_components')}, "
                f"cluster_id_groups={status.get('n_clusters_from_cluster_id')}"
            )
        else:
            print(
                f"{name}: error, single_agglomerate="
                f"{status.get('single_agglomerate')}, {status.get('error')}"
            )
    keys = ["case", "case_id", "particle_diameter_requested", "n_particles_requested",
            "morphology_requested", "fractal_dimension_requested",
            "fractal_prefactor_requested", "realization", "status",
            "post_vtk_directory", "contact_vtk_directory",
            "n_post_timesteps", "n_contact_timesteps", "n_common_timesteps",
            "first_common_timestep", "last_common_timestep",
            "selected_timestep", "converged", "cluster_id_available", "cluster_id_field",
            "n_clusters_from_cluster_id", "n_contact_components",
            "largest_component_fraction", "single_agglomerate",
            "n_particles", "n_contacts", "d_ref",
            "radius_of_gyration", "mean_coordination", "surface_fraction", "mean_graph_depth",
            "shielded_fraction_h_ge_2", "accessibility_score",
            "mean_geometric_exposure", "minimum_particle_geometric_exposure",
            "p10_particle_geometric_exposure", "maximum_particle_geometric_exposure",
            "exposure_samples_per_particle", "exposure_probe_radius_over_dref", "error"]
    with (outroot/"dataset_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    connected = sum(row.get("single_agglomerate") is True for row in rows)
    fragmented = sum(row.get("single_agglomerate") is False for row in rows)
    connectivity_unchecked = len(rows) - connected - fragmented
    relaxed = sum(row.get("converged") is True for row in rows)
    not_relaxed = sum(row.get("converged") is False for row in rows)
    print(
        "Connectivity summary: "
        f"single agglomerates={connected}, fragmented={fragmented}, "
        f"unchecked={connectivity_unchecked}"
    )
    print(f"Relaxation summary: converged={relaxed}, not converged={not_relaxed}")
    print(f"Manifest: {outroot/'dataset_manifest.csv'}")


if __name__ == "__main__":
    main()
