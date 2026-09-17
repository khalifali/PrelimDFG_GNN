import numpy as np
import sys
from pathlib import Path


ATOM_TYPES = 2
BOND_TYPES = 0
ANGLE_TYPES = 0
#bounds in x,y,z
BOUNDS = [(-100, 100), (-100, 100), (-100, 100)]

def writeLightsAgglomerateFile(outFile, particles, density):
    print("Writing agglomerate file to ", outFile, " length = ", particles.size)
    print()
    
    with open(outFile, "w") as f:
        
        ##Create Header
        #-> write "LIGGGHTS Description"
        f.write("LIGGGHTS Description\n\n")
        #-> write {particles.lenght} atoms
        f.write(f"{len(particles)} atoms\n")
        #-> write {atomTypes = 2} atom types
        f.write(f"{ATOM_TYPES} atom types\n")
        #-> write {bondTypes = 0} bond types
        f.write(f"{BOND_TYPES} bond types\n")
        #-> write {angleTypes = 0} angle types
        f.write(f"{ANGLE_TYPES} angle types\n\n")
        
        ##Write bounds
        f.write(f"{BOUNDS[0][0]}e00 {BOUNDS[0][1]}e00  xlo xhi\n")
        f.write(f"{BOUNDS[0][0]}e00 {BOUNDS[0][1]}e00  ylo yhi\n")
        f.write(f"{BOUNDS[0][0]}e00 {BOUNDS[0][1]}e00  zlo zhi\n\n")
        
        ##Write actual Atom content with indent
        f.write("Atoms\n\n")
        atomType = 1
        for atom_id, p in enumerate(particles, start=1):
            indent = " " * (6-len(str(atom_id)))
            diameter = 2 * p[3]
            f.write(indent + f"{atom_id}  {atomType} {diameter: .12E} {density: .12E} {p[0]: .12E} {p[1]: .12E} {p[2]: .12E}\n")
            
        f.write("\n")
            
        ##Write Velocities (all to 0 for now)
        f.write("Velocities\n\n")
        velocity = 0.0
        for atom_id, p in enumerate(particles, start=1):
            indent = " " * (6-len(str(atom_id)))
            f.write(indent + f"{atom_id} {velocity: .1f} {velocity: .1f} {velocity: .1f} {velocity: .1f} {velocity: .1f} {velocity: .1f}\n")
        
        #i dont know if this is needed, but the original file has it
        f.write("\n")
        
    
    
    

if len(sys.argv) < 2:
    print("Usage: python liggghtsWriter.py <input.csv>")
    sys.exit(1)

input_file = Path(sys.argv[1])

if not input_file.exists():
    raise FileNotFoundError(f"Input file does not exist: {input_file}")

particles = np.loadtxt(input_file, delimiter=",")

if particles.shape[1] < 4:
    raise ValueError("CSV file must contain at least four columns: x, y, z, radius")

output_file = input_file.with_suffix(".sys")

writeLightsAgglomerateFile(output_file, particles, 2.0)