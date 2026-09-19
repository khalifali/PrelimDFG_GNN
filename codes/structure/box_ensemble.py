"""Candidate v2 finite-scale estimator; not yet the production descriptor.

Uniform random rotations and independent 3-D grid shifts avoid PCA degeneracy.
The fitting range uses the rotation-invariant maximum centre distance + d.
Its changed range is explicit: values must not be mixed with legacy descriptors.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from descriptors import count_boxes


def box_ensemble(points, radius=.5, orientations=16, shifts=4, seed=38191):
    p = np.asarray(points, float)
    p = p-p.mean(0)
    d = 2*radius
    if orientations < 4 or orientations % 2 or shifts < 2:
        raise ValueError('Use even orientations >=4 and shifts >=2')
    extent = (float(pdist(p).max()) if len(p)>1 else 0.)+d
    upper = extent/4
    if upper <= d:
        return dict(box_dimension=None, box_quality='no_aggregate_scale_interval'), []
    sizes = np.geomspace(d, upper, 12)
    x = -np.log(sizes/d)
    rng = np.random.default_rng(seed)
    slopes=[]; trimmed=[]; r2=[]; curves=[]; local=[]
    for i in range(orientations):
        q, r = np.linalg.qr(rng.normal(size=(3,3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q)<0: q[:,0]*=-1
        pp=p@q
        for j in range(shifts):
            shift=rng.uniform(0,1,3)
            counts=[count_boxes(pp,radius,s,-shift*s) for s in sizes]
            y=np.log(counts); fit=linregress(x,y)
            slopes.append(fit.slope);r2.append(fit.rvalue**2)
            trimmed.extend([linregress(x[1:],y[1:]).slope,linregress(x[:-1],y[:-1]).slope])
            ls=np.diff(y)/np.diff(x);local.append(ls)
            curves.extend(dict(orientation=i,grid=j,size_over_d=float(s/d),count=int(c)) for s,c in zip(sizes,counts))
    mean=float(np.mean(slopes)); spread=float(np.std(slopes,ddof=1))
    half=len(slopes)//2
    convergence=float(abs(np.mean(slopes[:half])-np.mean(slopes[half:])))
    sensitivity=float(max(abs(np.asarray(trimmed)-mean)))
    span=float(np.log10(upper/d)); reasons=[]
    if span<.5: reasons.append('scale_span_below_half_decade')
    if min(r2)<.95: reasons.append('fit_R2_below_0.95')
    if spread>.10: reasons.append('grid_SD_above_0.10')
    if convergence>.03: reasons.append('ensemble_half_difference_above_0.03')
    if sensitivity>.20: reasons.append('window_sensitivity_above_0.20')
    return dict(box_dimension=mean,box_quality='ok' if not reasons else ';'.join(reasons),
        box_grid_sd=spread,box_min_fit_r2=float(min(r2)),box_scale_span_decades=span,
        box_window_sensitivity=sensitivity,box_ensemble_half_difference=convergence,
        box_mean_local_slopes=np.mean(local,axis=0).tolist(),
        estimator='ensemble_v2_candidate',orientations=orientations,shifts=shifts,seed=seed),curves
