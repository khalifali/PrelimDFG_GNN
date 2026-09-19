# Geometric exposure and shielding: literature motivation

Updated 19 September 2026. Purpose: support the motivation and discussion of the agglomerate graph-learning paper. This is a focused reading note, not a systematic review.

## Scope and central distinction

Particle shielding has established uses in aggregate heat/mass transfer, friction and mobility, and flow-induced internal stresses. These applications motivate studying how surrounding particles screen individual particle surfaces.

The present target is narrower: for each surface sample on a spherical primary particle, the code tests whether the outward-normal ray intersects another particle. The fraction of unobstructed samples is a_geo,i; its particle average is the present monodisperse aggregate target. It is a geometric normal-ray visibility measure. It is not automatically fluid-accessible area, a gas collision probability, a radiative view factor, a drag correction, or a transport effectiveness factor.

No exact precedent for this particular normal-ray definition has been verified in the references below. Literature relevance establishes the motivation for studying screening; it does not calibrate this target against physical transfer.

Breakage references below provide application context only. Breakage modelling and prediction remain outside this paper's scope, to preserve separation from the ongoing DFG deagglomeration project.

## Heat transfer, mass transfer and surface access

### Filippov, Zurita and Rosner (2000)

**Fractal-like Aggregates: Relation between Morphology and Physical Properties.** Journal of Colloid and Interface Science 229, 261–273.
[Publisher](https://www.sciencedirect.com/science/article/pii/S0021979700970279) · [DOI](https://doi.org/10.1006/jcis.2000.7027)

The authors numerically investigate mass/energy exchange between spherical-particle aggregates and gas in free-molecular and continuum limits, together with optical properties. Laser-induced incandescence and the heating/cooling needed for soot sizing provide an engineering application. Their results relate morphology to effective transport dimensions.

**Use in our motivation:** aggregate arrangement changes exchange with the environment, making independent isolated-particle descriptions insufficient. **Limit:** their physical transport quantities must not be identified with our normal-ray surface fraction. This annotation is based on the accessible abstract; detailed transport equations should be checked in the full paper before reproducing a correlation.

### Coelho, Bekki, Thovert and Adler (2000)

**Uptake on fractal particles: 1. Theoretical framework.** Journal of Geophysical Research: Atmospheres 105(D3), 3905–3916.
[DOI](https://doi.org/10.1029/1999JD900815) · [Accessible paper copy](https://www.academia.edu/17846221/Uptake_on_fractal_particles_1_Theoretical_framework)

The study solves steady diffusion with a surface uptake condition on numerically generated fractal particles. In the reaction-limited regime the uptake depends on fluid-accessible surface area; in the diffusion-limited regime an effective diffusion radius describes the total uptake. Closed internal cavities are distinguished from surfaces accessible from the external fluid.

**Use:** surface access matters for heterogeneous uptake, while transport resistance determines whether the entire available surface contributes effectively. **Limit:** accessibility through connected fluid paths is different from straight-line visibility. These calculations use cubic constituent elements; do not describe them as identical to our spherical-particle geometry.

## Drag, mobility and individual-particle shielding

### Isella and Drossinos (2011)

**On the friction coefficient of straight-chain aggregates.** Journal of Colloid and Interface Science 356, 505–513.
[Author manuscript](https://arxiv.org/html/1007.2801v2) · [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0021979711001032)

The proposed collision-rate method obtains molecule–aggregate collision rates from a diffusion equation with an absorbing aggregate boundary. Individual and aggregate-average monomer shielding factors are connected to friction coefficients. The authors also use the appropriately shielded friction/random-force description in Langevin simulations of aggregate diffusion.

**Use:** a particularly direct precedent for predicting local shielding as well as an aggregate average. **Limit:** these are diffusion-derived factors linked through an approximate friction methodology, not binary geometric ray visibility. Do not present the diffusion–friction relationship as a universal identity.

### Melas, Isella, Konstandopoulos and Drossinos (2014)

**Morphology and mobility of synthetic colloidal aggregates.** Journal of Colloid and Interface Science 417, 27–36.
[Author manuscript](https://arxiv.org/abs/1311.2802)

The authors connect fractal-like aggregate geometry to hydrodynamic radius and dynamic shape factors in the continuum regime, using collision rates obtained from a Laplace problem. Individual monomer hydrodynamic shielding factors are calculated for DLCA aggregates. They also propose a geometric correlation using radius of gyration and primary-particle count.

**Use:** supports both local shielding analysis and strong conventional-descriptor baselines. **Limit:** a graph model's advantage cannot be presumed; a useful global correlation may already capture much of the response.

## Breakage: hydrodynamic interactions, screening and internal load transmission

These studies address a related physical consequence of the surrounding structure. Their role here is motivation, not a new breakage task.

### Ó Conchúir and Zaccone (2013)

**Mechanism of flow-induced biomolecular and colloidal aggregate breakup.** Physical Review E 87, 032310.
[Publisher and abstract](https://link.aps.org/doi/10.1103/PhysRevE.87.032310) · [Full-text author-uploaded copy](https://www.researchgate.net/publication/258080484_Mechanism_of_flow-induced_biomolecular_and_colloidal_aggregate_breakup)

A drift–diffusion description of bond dissociation is extended to aggregates using a semiempirical stress-transmission efficiency Gamma. The full-text discussion connects aggregate structure with many-body stress transmission and hydrodynamic screening. The theory distinguishes collective breakup from surface erosion, with dependence on size and fractal structure.

**Use:** explicit support for screening and stress transmission in breakup reasoning. **Limit:** Gamma is not our exposure fraction and is not a purely geometric visibility factor. Although the title includes biomolecules, the framework also concerns colloidal aggregates; it need not make our paper biologically oriented.

### De Bona, Lanotte and Vanni (2014)

**Internal stresses and breakup of rigid isostatic aggregates in homogeneous and isotropic turbulence.** Journal of Fluid Mechanics 755.
[DOI](https://doi.org/10.1017/jfm.2014.421) · [Author manuscript](https://arxiv.org/abs/1405.3704)

The study combines turbulence DNS with a discrete-element treatment based on Stokesian dynamics. It computes internal loading while aggregates move in turbulence and estimates breakup rates and fragment distributions. Failure occurs when a contact tensile force exceeds its cohesive strength; the examined isostatic structures can disconnect after a single contact fails.

**Use:** demonstrates why particle-level hydrodynamic loading and its transmission through the contact network matter. **Limit:** this is a hydrodynamic-interaction calculation, not evidence that a scalar ray exposure reproduces those forces or predicts failure.

### Harshe and Lattuada (2012): additional reading

**Breakage Rate of Colloidal Aggregates in Shear Flow through Stokesian Dynamics.** Langmuir 28, 283–292.
[Publisher/DOI](https://doi.org/10.1021/la2038476)

A relevant Stokesian-dynamics breakage reference. Bibliographic identity is verified, but full publisher text was not accessible in this review. Read the original before attributing a particular shielding formula, quantitative correction or performance result to it. The publication appeared online in 2011; the journal-volume citation uses 2012.

## How to use the literature without conflating quantities

| Quantity | What it measures | Relation to the present target |
|---|---|---|
| Normal-ray geometric exposure | Fraction of surface normals with an unobstructed ray | Current target |
| Fluid-accessible surface area | Surface reachable through connected fluid space | May remain accessible despite a blocked normal ray |
| Diffusive shielding | Reduction of uptake due to the surrounding diffusion field | Requires diffusion physics or a validated surrogate |
| Hydrodynamic shielding | Modification of particle loading through fluid interactions | Depends on flow and the force/torque definition |
| Stress-transmission efficiency | Collective loading transmitted through an aggregate | Also involves connectivity and mechanical assumptions |
| Radiative view factor | Angularly weighted exchange between surfaces | A single direction per surface point does not supply the angular integral |

For non-overlapping spheres, point contacts remove no finite surface area, yet neighbours can still block outward rays and modify transport fields. This is why the word “exposed” must always be accompanied by its operational definition.

The following is a conceptual chain, not an established correlation for this dataset:
particle arrangement → geometric screening and fluid interactions → local exchange/forces → aggregate-scale response.

## Suggested manuscript wording

The literature on aggregate transport demonstrates that neighbouring primary particles modify exchange with the surrounding medium. Morphology-dependent mass and energy transfer have been investigated for fractal-like aerosols (Filippov et al., 2000), while surface access and diffusion resistance jointly determine gas uptake on fractal particles (Coelho et al., 2000). Individual-particle shielding has also been used in descriptions of aggregate friction and mobility (Isella and Drossinos, 2011; Melas et al., 2014). Related work on flow-induced breakup emphasizes the role of hydrodynamic interactions and internal stress transmission (Ó Conchúir and Zaccone, 2013; De Bona et al., 2014). These findings motivate studying spatial screening within agglomerates. Here, we isolate its geometric component through a precisely defined normal-ray exposure measure. This measure is investigated as a structural descriptor; quantitative equivalence to a transport coefficient or hydrodynamic shielding factor is not assumed.

## Claims supported and outstanding

Supported motivation: local surroundings influence physical interaction with the external environment, and aggregate averages can conceal particle-level heterogeneity.

Current demonstrable target: numerical geometric normal-ray exposure, with sampling/orientation sensitivity checks. Predicting it does not itself establish a physical surrogate.

Outstanding if stronger claims are desired: compare against a specifically chosen literature-defined shielding quantity in a specified regime; measure end-to-end runtime if claiming speed; establish model advantage over strong local/global descriptor baselines and generalization to fresh structures. No additional physical simulations or breakage campaign are authorized by this documentation update.

Box-counting fractal dimension and particle-body convex-hull porosity remain the required morphology descriptors. Literature mass–radius fractal exponents should not be silently equated with finite-scale box-counting estimates.
