#!/usr/bin/env python3

import argparse
from pathlib import Path


def read_liggghts_dump(filename):
    frames = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        timestep = int(lines[i + 1].strip())

        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError(f"Unexpected format in {filename}")

        n_atoms = int(lines[i + 3].strip())

        if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError(f"Unexpected format in {filename}")

        box = []

        for j in range(3):
            lo, hi = map(float, lines[i + 5 + j].split()[:2])
            box.append((lo, hi))

        atom_header = lines[i + 8].strip().split()[2:]
        atom_lines = lines[i + 9:i + 9 + n_atoms]

        atoms = []

        for line in atom_lines:
            values = line.split()
            atom = dict(zip(atom_header, values))
            atoms.append(atom)

        frames.append((timestep, box, atom_header, atoms))

        i += 9 + n_atoms

    return frames


def get_float(atom, names, default=0.0):
    for name in names:
        if name in atom:
            return float(atom[name])
    return default


def get_int(atom, names, default=0):
    for name in names:
        if name in atom:
            return int(float(atom[name]))
    return default


def write_particles_vtk(filename, timestep, box, atoms):

    with open(filename, "w") as f:

        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"LIGGGHTS particles timestep {timestep}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {len(atoms)} float\n")

        for atom in atoms:

            x = get_float(atom, ["x", "xu", "xs"])
            y = get_float(atom, ["y", "yu", "ys"])
            z = get_float(atom, ["z", "zu", "zs"])

            if "xs" in atom:
                x = box[0][0] + x * (box[0][1] - box[0][0])

            if "ys" in atom:
                y = box[1][0] + y * (box[1][1] - box[1][0])

            if "zs" in atom:
                z = box[2][0] + z * (box[2][1] - box[2][0])

            f.write(f"{x} {y} {z}\n")

        f.write(f"\nVERTICES {len(atoms)} {2 * len(atoms)}\n")

        for i in range(len(atoms)):
            f.write(f"1 {i}\n")

        f.write(f"\nPOINT_DATA {len(atoms)}\n")

        # particle id
        f.write("SCALARS id int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for atom in atoms:
            f.write(f"{get_int(atom,['id'])}\n")

        # particle type
        f.write("SCALARS type int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for atom in atoms:
            f.write(f"{get_int(atom,['type'])}\n")

        # radius
        if any("radius" in atom for atom in atoms):

            f.write("SCALARS radius float 1\n")
            f.write("LOOKUP_TABLE default\n")

            for atom in atoms:
                f.write(f"{get_float(atom,['radius'])}\n")

        elif any("r" in atom for atom in atoms):

            f.write("SCALARS radius float 1\n")
            f.write("LOOKUP_TABLE default\n")

            for atom in atoms:
                f.write(f"{get_float(atom,['r'])}\n")

        # velocity
        if all(k in atoms[0] for k in ["vx", "vy", "vz"]):

            f.write("VECTORS velocity float\n")

            for atom in atoms:
                vx = get_float(atom, ["vx"])
                vy = get_float(atom, ["vy"])
                vz = get_float(atom, ["vz"])

                f.write(f"{vx} {vy} {vz}\n")


def write_box_vtk(filename, timestep, box):

    xlo, xhi = box[0]
    ylo, yhi = box[1]
    zlo, zhi = box[2]

    points = [
        (xlo, ylo, zlo),
        (xhi, ylo, zlo),
        (xhi, yhi, zlo),
        (xlo, yhi, zlo),
        (xlo, ylo, zhi),
        (xhi, ylo, zhi),
        (xhi, yhi, zhi),
        (xlo, yhi, zhi),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    with open(filename, "w") as f:

        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"LIGGGHTS box timestep {timestep}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write("POINTS 8 float\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        f.write(f"\nLINES {len(edges)} {len(edges) * 3}\n")

        for a, b in edges:
            f.write(f"2 {a} {b}\n")


def output_names(
    input_file: Path,
    timestep: int,
    frame_count: int,
    output_dir: Path,
):
    """Return particle and box VTK paths inside the output directory."""

    if frame_count == 1:
        particle_name = f"{input_file.stem}.vtk"
        box_name = f"{input_file.stem}_box.vtk"
    else:
        particle_name = f"{input_file.stem}_{timestep}.vtk"
        box_name = f"{input_file.stem}_{timestep}_box.vtk"

    return output_dir / particle_name, output_dir / box_name


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIGGGHTS post_*.txt particle dumps to VTK."
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default="post_*.txt",
        help="Input glob pattern (default: post_*.txt)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="post_vtk",
        help="Output directory (default: post_vtk)",
    )
    args = parser.parse_args()

    txt_files = sorted(Path(".").glob(args.pattern))

    if not txt_files:
        print(f"No files found for pattern: {args.pattern}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(txt_files)} particle dump files")

    for input_file in txt_files:

        print(f"\nProcessing {input_file}")

        try:

            frames = read_liggghts_dump(input_file)

            if not frames:
                print("  No dump frames found")
                continue

            for timestep, box, header, atoms in frames:
                particle_file, box_file = output_names(
                    input_file, timestep, len(frames), output_dir
                )

                write_particles_vtk(
                    particle_file,
                    timestep,
                    box,
                    atoms
                )

                write_box_vtk(
                    box_file,
                    timestep,
                    box
                )

                print(f"  particles -> {particle_file}")
                print(f"  box       -> {box_file}")

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
