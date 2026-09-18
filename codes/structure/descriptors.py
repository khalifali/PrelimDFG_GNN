"""Geometry-only descriptors for equal spheres; no requested Df/kf inputs.

The sparse box counter uses the sphere/cube intersection rule in the repository's
DimensionAnalysis/testdim.py. It avoids that script's clipping domain and global
subparticle-scale fit. Its output is a finite-scale exponent, not an asymptotic
dimension or an exact reproduction of the legacy estimator.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import linregress


def normalize(xyz, radii):
    xyz=np.asarray(xyz,float); radii=np.asarray(radii,float)
    if xyz.shape!=(len(radii),3) or not np.isfinite(xyz).all() or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('Invalid geometry')
    if not np.allclose(radii,radii[0],rtol=1e-8,atol=0):
        raise ValueError('Body-hull formula currently requires monodisperse spheres')
    # Quantization below any physical tolerance avoids a scale/translation-only
    # roundoff change selecting different PCA axes for symmetric configurations.
    return np.round((xyz-xyz.mean(axis=0))/(2*radii[0]),12), np.full(len(radii),.5)


def body_hull_volume(points, radius):
    """Exact parallel-body volume conv(centres) + radius*unit_ball.

For rank 3 use Steiner's formula V+A*r+M*r²+4*pi*r³/3, where
M=1/2 sum(edge_length * exterior_dihedral_angle). Triangulation diagonals
have zero angle. Rank 0/1/2 cases use sphere/capsule/rounded planar hull.
"""
    p=np.asarray(points,float); p=p-p.mean(axis=0)
    _,s,vt=np.linalg.svd(p,full_matrices=False)
    rank=int(np.sum(s>max(float(s[0]) if len(s) else 0.,1.)*1e-10))
    ball=4*np.pi*radius**3/3
    if rank==0:return ball
    projected=p@vt[:rank].T
    if rank==1:return np.ptp(projected[:,0])*np.pi*radius**2+ball
    if rank==2:
        h=ConvexHull(projected)
        return 2*radius*h.volume+.5*np.pi*radius**2*h.area+ball
    h=ConvexHull(p)
    normals=h.equations[:,:3]; normals/=np.linalg.norm(normals,axis=1)[:,None]
    edges={}
    for face, triangle in enumerate(h.simplices):
        for a,b in itertools.combinations(triangle,2):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(face)
    curvature=0.
    for (a,b),faces in edges.items():
        if len(faces)!=2:raise ValueError('Nonmanifold convex hull')
        cosine=float(np.clip(normals[faces[0]]@normals[faces[1]],-1,1))
        angle=0. if cosine>1-1e-12 else np.arccos(cosine)
        curvature+=.5*np.linalg.norm(p[a]-p[b])*angle
    return float(h.volume+radius*h.area+curvature*radius**2+ball)


def hull_porosity(points, radius=.5):
    volume=body_hull_volume(points,radius)
    solid=len(points)*4*np.pi*radius**3/3
    distances=pdist(points)
    overlap=distances[distances<2*radius]
    # Sum of pairwise lens volumes bounds total overcount of sum(sphere volumes),
    # including when triple overlaps exist (Bonferroni bound).
    correction=float(np.sum(np.pi*(4*radius+overlap)*(2*radius-overlap)**2/12))
    error=correction/volume
    porosity=1-solid/volume
    if error>1e-5 or porosity < -1e-8:
        raise ValueError('Overlaps too large for the documented solid-volume approximation')
    return dict(hull_porosity=float(porosity), hull_volume_over_d3=float(volume),
                porosity_overlap_error_bound=float(error))


def count_boxes(points, radius, size, origin):
    """Sparse, vectorized count of boxes intersecting the closed sphere volumes."""
    if size<2*radius*(1-1e-12):raise ValueError('This aggregate-scale counter requires box size >= diameter')
    offsets=np.array(list(itertools.product((-1,0,1),repeat=3)))
    base=np.floor((points-origin)/size).astype(np.int64)
    indices=base[:,None,:]+offsets[None,:,:]
    lo=origin+indices*size
    closest=np.maximum(lo,np.minimum(points[:,None,:],lo+size))
    occupied=np.sum((closest-points[:,None,:])**2,axis=2)<=radius**2*(1+1e-12)
    return len(np.unique(indices[occupied],axis=0))


def box_dimension(points, radius=.5):
    # Principal-axis frame reduces arbitrary input orientation dependence.
    centered=points-points.mean(axis=0)
    _,_,vt=np.linalg.svd(centered,full_matrices=False)
    canonical=centered@vt.T
    for k in range(3):
        i=np.argmax(np.abs(canonical[:,k]))
        if canonical[i,k]<0:canonical[:,k]*=-1
    diameter=2*radius
    extent=float(np.max(np.ptp(canonical,axis=0))+diameter)
    largest=extent/4
    if largest<=diameter:
        raise ValueError('No aggregate-scale fitting interval between d and L/4')
    sizes=np.geomspace(diameter,largest,8)
    rotations=[np.eye(3)]
    rng=np.random.default_rng(38191)
    for _ in range(2):
        q,_=np.linalg.qr(rng.normal(size=(3,3)));rotations.append(q)
    curves=[]; slopes=[]; r2=[]; stderr=[]; window=[]
    for orientation,rotation in enumerate(rotations):
        pp=canonical@rotation
        for shift in (0.,.5):
            counts=[]
            for size in sizes:
                origin=pp.min(axis=0)-radius-shift*size
                count=count_boxes(pp,radius,size,origin); counts.append(count)
                curves.append(dict(orientation=orientation,shift=shift,box_size_over_d=float(size/diameter),occupied=count))
            x=np.log(diameter/sizes);y=np.log(counts)
            fit=linregress(x,y)
            slopes.append(fit.slope);r2.append(fit.rvalue**2);stderr.append(fit.stderr)
            window.extend([linregress(x[1:],y[1:]).slope,linregress(x[:-1],y[:-1]).slope])
    mean=float(np.mean(slopes));span=float(np.log10(largest/diameter))
    spread=float(np.std(slopes,ddof=1));sensitivity=float(max(abs(np.asarray(window)-mean)))
    # Flags are descriptor diagnostics, not post-hoc case exclusions.
    reasons=[]
    if span<.5:reasons.append('scale_span_below_half_decade')
    if min(r2)<.95:reasons.append('fit_R2_below_0.95')
    if spread>.10:reasons.append('grid_SD_above_0.10')
    if sensitivity>.20:reasons.append('window_sensitivity_above_0.20')
    return dict(box_dimension=mean,box_grid_sd=spread,box_min_fit_r2=float(min(r2)),
        box_mean_slope_stderr=float(np.mean(stderr)),box_scale_span_decades=span,
        box_window_sensitivity=sensitivity,box_quality='ok' if not reasons else ';'.join(reasons)),curves


def measure(xyz,radii):
    points,r=normalize(xyz,radii)
    box,curves=box_dimension(points,r[0])
    return dict(**box,**hull_porosity(points,r[0])),curves
