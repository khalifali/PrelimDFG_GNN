#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 12 11:36:48 2026

@author: sp00feR

There are a lot of optimization strategies that can still be applied to this code but this will only be needed for agglomerates exceeding 10k particles i think.

Based on this: 
Krzysztof Skorupski, Janusz Mroczka, Thomas Wriedt, Norbert Riefler,
A fast and accurate implementation of tunable algorithms used for generation of fractal-like aggregate models,
Physica A: Statistical Mechanics and its Applications,
Volume 404,
2014,
Pages 106-117,
ISSN 0378-4371,
https://doi.org/10.1016/j.physa.2014.02.072.
(https://www.sciencedirect.com/science/article/pii/S0378437114001812)
Abstract: In many branches of science experiments are expensive, require specialist equipment or are very time consuming. Studying the light scattering phenomenon by fractal aggregates can serve as an example. Light scattering simulations can overcome these problems and provide us with theoretical, additional data which complete our study. For this reason a fractal-like aggregate model as well as fast aggregation codes are needed. Until now various computer models, that try to mimic the physics behind this phenomenon, have been developed. However, their implementations are mostly based on a trial-and-error procedure. Such approach is very time consuming and the morphological parameters of resulting aggregates are not exact because the postconditions (e.g. the position error) cannot be very strict. In this paper we present a very fast and accurate implementation of a tunable aggregation algorithm based on the work of Filippov et al. (2000). Randomization is reduced to its necessary minimum (our technique can be more than 1000 times faster than standard algorithms) and the position of a new particle, or a cluster, is calculated with algebraic methods. Therefore, the postconditions can be extremely strict and the resulting errors negligible (e.g. the position error can be recognized as non-existent). In our paper two different methods, which are based on the particle–cluster (PC) and the cluster–cluster (CC) aggregation processes, are presented.
Keywords: Fractal aggregates; Particle–cluster aggregation; Cluster–cluster aggregation
"""

import numpy as np
import math
import pyvista as pv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import sys

""" #CONFIGURATION
particle_radius = 1.0
random_seed = 420
kf = 0.4
Df = 3
N_target = 1000
max_try = 1000 """

#just a basic dataclass to make config file loading easier and have a nice debug print method
@dataclass
class Config:
    particle_radius: float
    random_seed: int
    kf: float
    Df: float
    N_target: int
    max_try: int
    def debug_print(self) -> None:
        print("=" * 40)
        print("CONFIG DEBUG")
        print("=" * 40)
        print(f"particle_radius : {self.particle_radius:>10.4f}")
        print(f"random_seed     : {self.random_seed:>10}")
        print(f"kf              : {self.kf:>10.4f}")
        print(f"Df              : {self.Df:>10.4f}")
        print(f"N_target        : {self.N_target:>10}")
        print(f"max_try         : {self.max_try:>10}")
        print("=" * 40)

#simple XML parser for loading config data into a Config dataclass object, this is especially usefull for batch tests
def loadXMLConfig(xml_path: str) -> Config:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get_text(tag: str) -> str:
        elem = root.find(tag)
        if elem is None or elem.text is None:
            raise ValueError(f"Missing XML-Tag: {tag}")
        return elem.text.strip()

    return Config(
        particle_radius=float(get_text("particle_radius")),
        random_seed=int(get_text("random_seed")),
        kf=float(get_text("kf")),
        Df=float(get_text("Df")),
        N_target=int(get_text("N_target")),
        max_try=int(get_text("max_try")),
    )

#basic vector normalization method
def normalize(v):
    return v / np.linalg.norm(v)

#this is used for creating a vector perpendicular to v
def createPerpendicularVector(v):
    if abs(v[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])

    perpendicular = np.cross(v, helper)
    perpendicular = normalize(perpendicular)
    return perpendicular

#this is used for rotating a vector around an axis using Rodrigues rotation formula
def rotateVector(v, axis, angle):
    axis = normalize(axis)
    rotated_v = (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )
    return rotated_v

#this is used for checking if the new particle overlaps with any particle in LA
def checkOverlap(config, new_particle, LA):
    for q in range(len(LA)):
        #calculate distance between new particle and particle q from candidate list
        d = np.linalg.norm(new_particle - LA[q])
        #check if particles overlap
        if d < 2 * config.particle_radius:
            return True
    return False


#this is Eq. 16, Marsaglia Method
def randomSpherePoint():
    while True:
        #create two random numbers from -1 to 1 
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)

        #check wether criterium x1² + x2² < 1
        c = x1**2 + x2**2

        if c < 1:
            #create R vector and leave
            R = np.array([
                2*x1*np.sqrt(1-c),
                2*x2*np.sqrt(1-c),
                1 - 2*c
            ])
            return R

#this is used for calculating the geometric center of an aggregate according to Eq. 17a
def calculateCenter(particles):
    sum_P = 0

    for q in range(len(particles)):
        sum_P += particles[q]

    C_Np = sum_P / len(particles)
    return C_Np

#this is used for updating the geometric center incrementally according to Eq. 17b
def calculateCenterInc(center, N_old, new_particle):
    C_new = (center * N_old + new_particle) / (N_old + 1)
    return C_new

#this creates a list of candidates for the addition of a new particle "the distance between the centers of the particles must be equal or greater than 2rp"
def createCandidateList(config, particles, center, gamma):
    LA = []
    for q in range(len(particles)):
        #calculate distance of particle q to geometric center
        d = np.linalg.norm(particles[q] - center)
        #only particles close enough to the outer region can be reference particles
        if d >= gamma - 2 * config.particle_radius:
            LA.append(particles[q])
    return LA

#this calculates alpha according to Appendix A3  Eq. 18
def calculateAlpha(config, center, reference_particle, gamma):
    CB = reference_particle - center
    CB_length = np.linalg.norm(CB)
    AB = 2 * config.particle_radius

    cos_alpha = (gamma**2 + CB_length**2 - AB**2) / (2 * gamma * CB_length)

    alpha = math.acos(cos_alpha)
    return alpha

#this creates a new particle based on center, gamma and a ref particle
def createNewParticle(config, center, gamma, reference_particle):
    #horrible geometrical shit here
    #we are searchging for a point A, where A is on the Particle around C with r = gamma && |AB| = 2rp
    #- calculate alpha according to Appendix A3  Eq. 18
    alpha = calculateAlpha(config, center, reference_particle, gamma)
    #- create vector from center to ref_particle
    CB = reference_particle - center
    
    #- create perpendicular unit vector 
    CB_unit = normalize(CB)
    perpendicular = createPerpendicularVector(CB_unit)
    
    #- create gamma vector with angle alpha to CB
    gamma_vector = gamma * (math.cos(alpha) * CB_unit + math.sin(alpha) * perpendicular)
    
    #- choose rabdom rotation angle beta
    beta = np.random.uniform(0, 2 * math.pi)
    
    #- rotate gamma vector around CB axis
    gamma_vector_rotated = rotateVector(gamma_vector, CB_unit, beta)
    
    #- calculate new particle position
    new_particle = center + gamma_vector_rotated
    
    return new_particle

#read cmdline arg
if len(sys.argv) < 2:
    print("Usage: python PC.py <configFile>")
    sys.exit(1)

config_file = sys.argv[1]

#generate config from XML
config = loadXMLConfig(config_file)
config.debug_print()
#set a fixed random seed for reproducibility
np.random.seed(config.random_seed)

#1. create two contacting particles
particles = []

# first particle at geometry origin 0,0,0
p1 = np.array([0.0, 0.0, 0.0])
particles.append(p1)

#add second particle in direct contact
d = randomSpherePoint()
p2 = p1 + d * (2 * config.particle_radius)
particles.append(p2)

#debug
print(particles)
print("Distance:", np.linalg.norm(particles[1] - particles[0]))


#main loop until target number of particles is reached
while len(particles) < config.N_target:

    #2. calculate geometric center of the cluster only for the first 2, after we do it incrementally
    if(len(particles) == 2):
        center = calculateCenter(particles)

    #3. calculate gamma according to Eq. 2 -> radius where the new particle has to be 
    Np = len(particles) + 1

    gamma_squared = (
        ((Np**2 * config.particle_radius**2) / (Np - 1)) * ((Np / config.kf) ** (2 / config.Df))
        - ((Np * config.particle_radius**2) / (Np - 1))
        - (Np * config.particle_radius**2) * (((Np - 1) / config.kf) ** (2 / config.Df))
    )

    gamma = math.sqrt(gamma_squared)

    #print("Current particles:", len(particles))
    #print("Center:", center)
    #print("Gamma:", gamma)
    
    #4. create candidate list LA
    LA = createCandidateList(config, particles, center, gamma)
    #print(LA)
    #5. try to find one valid new particle
    particle_added = False
    tr = 0
    while particle_added == False:
       tr += 1
       #choose a random reference particle (is this right?? idk i am brain damaged)
       reference_particle = LA[np.random.randint(len(LA))]
       #print(reference_particle)
   
       #6. create new particle
       new_particle = createNewParticle(config, center, gamma, reference_particle)

       #7. check overlap
       overlap = checkOverlap(config, new_particle, LA)
       #8. append particle if no overlap exists
       if overlap == False:
            N_old = len(particles)
            particles.append(new_particle)
            center = calculateCenterInc(center, N_old, new_particle)
            particle_added = True
            print("Particle added (", len(particles), "/", config.N_target, ")")

       if tr > config.max_try:
           raise Exception("Exeedet maximum number of trys for adding particle. Check generation Parameters!!")
           

#save as CSV
name = "aggr_Df_" + str(config.Df) + "_kf_" +str(config.kf) + "_N_" +str(config.N_target) +".csv"
#add particle radius to the output particle list
particles_out = np.column_stack((particles, np.full(len(particles), config.particle_radius)))
#export to csv 
np.savetxt(name, particles_out, delimiter=",")


#Visualization using glyphs, it looks nice and is useful for understanding the effects of the generation params

""" points = np.array(particles)

cloud = pv.PolyData(points)
cloud["radius"] = np.full(len(points), config.particle_radius)

sphere = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)

glyphs = cloud.glyph(scale="radius", geom=sphere)

plotter = pv.Plotter()
plotter.add_mesh(glyphs, smooth_shading=True)
plotter.show() """