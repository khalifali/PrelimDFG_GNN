#!/usr/bin/env python3
"""Surface-normal line-of-sight exposure, not diffusion or oxygen uptake."""
import numpy as np


def directions(count):
    if count < 16:raise ValueError('Use at least 16 surface directions')
    k=np.arange(count,dtype=float)
    z=1-2*(k+.5)/count
    angle=k*np.pi*(3-np.sqrt(5))
    rr=np.sqrt(1-z*z)
    return np.column_stack([rr*np.cos(angle),rr*np.sin(angle),z])


def geometric_exposure(xyz,radii,rays=512):
    xyz=np.asarray(xyz,dtype=float);radii=np.asarray(radii,dtype=float)
    if xyz.shape!=(len(radii),3) or not np.isfinite(xyz).all() or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('Finite coordinates and positive particle radii required')
    # Scale geometry for roundoff control. Result is independent of length units.
    scale=np.median(radii);xyz=(xyz-xyz.mean(axis=0))/scale;radii=radii/scale
    dirs=directions(rays)
    result=np.empty(len(xyz))
    for i in range(len(xyz)):
        others=np.arange(len(xyz))!=i
        delta=xyz[others]-xyz[i];r2=radii[others]**2
        d2=np.sum(delta*delta,axis=1)
        blocked_count=0
        # Process 128 rays at a time to bound working memory for larger aggregates.
        for start in range(0,rays,128):
            projection=delta@dirs[start:start+128].T
            perpendicular2=np.maximum(0,d2[:,None]-projection**2)
            disc=r2[:,None]-perpendicular2
            # The forward intersection must lie beyond this particle's surface.
            far=projection+np.sqrt(np.maximum(0,disc))
            hit=(disc>=0)&(far>=radii[i]*(1+1e-8))
            blocked_count+=np.any(hit,axis=0).sum()
        result[i]=1-blocked_count/rays
    return result
