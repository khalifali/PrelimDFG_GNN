# Matched BPM–vdW campaign and later CFD–DEM use

Active workflow: Matched vdW relaxation and ML. BPM training is not cancelled,
modified or overwritten. Its complete v9 reference is run 35307361168. The vdW
arm reuses exactly the selected 179 initial geometries and their initial
velocities/spins from BPM run 35290323430. Failed BPM case001 is NOT the exclusion:
the existing exclusion is fractal_dp2_N0100_Df2.6_kf0.8_rep04, kept unchanged.

Match material E, nu, restitution, friction, mass, numerical translational and
rotational drag, initial kicks, box and boundaries, dt=1e-10 s, duration=16 us,
three output times and 2048 Fibonacci rays. Remove permanent bonds and overlay
Hertz/Mindlin with the unmodified pinned LAMFOAM lamfoam/vdw law. A=3.028e-20 J,
h0=4e-10 m come from the supplied historical input. This isolates the mechanical
model under the current campaign conditions; it is NOT a reproduction of all
old LIGGGHTS rolling/torsion/history models or its different nu/restitution.

Each case verifies all initial coordinates, radii, velocities and spins against
BPM before acceptance. vdW can release attractive potential energy, so the BPM
KE/initial-KE ratio is not reused as an equilibrium certificate. Require finite
outputs, preserved particle count, and max overlap/d<.001. Report KE, geometry
and exposure at all three times. Any failure stops the paired experiment; no
additional vdW cases are silently dropped. Results are a fixed-time diagnostic;
strong midpoint-to-final changes require longer relaxation before physical
interpretation. Numerical timestep convergence is still needed before treating
small model differences as physical. No simulation is labelled equilibrium just
because LAMMPS exited successfully.

Train core134 and extended179 with the same v9 grouping, seeds, q=1..4, settings
and explicit byte-for-byte split verification against the saved BPM splits.
Compare GNN, ANN, linear and forests. Geometry comparison reports changes from
the common initial state and between final models in contacts, components,
coordination, Rg, particle displacement and exposure. Distinguish permanent
mechanical bonds from geometric contacts used by the GNN.

For future CFD–DEM, identical initialization is guaranteed; identical relaxed
structure is not. BPM constrains a chosen reference network while vdW can alter
it. State tolerances for structural equivalence (coordinates, contacts, Rg,
exposure) before claiming equivalence; do not move coordinates or fit parameters
post hoc solely to match an ML score. Dynamic mechanical equivalence requires
additional response/load tests, not just one relaxed shape.

The original uploaded case001 SYS/CSV are preserved as an additional diagnostic;
they have identical coordinates. They are not substituted into the reconstructed
179-case training dataset. The original-case comparison script is separate and
is not launched by the matched campaign.
