"""Tunable particle/cluster aggregation with exact mass-radius constraints.

Reconstruction of the formulation described in the old report, not recovered
historical source. Internal primary radius is one; solid-sphere Rg includes 3/5.
"""
import numpy as np
from scipy.spatial import cKDTree
from generate import rotation, inspect


def rg2(p):
    return np.mean(np.sum((p-p.mean(axis=0))**2,axis=1))+.6


def circle_points(center, radius, rng, count=24):
    """Translations t satisfying |t|=radius and |t-center|=2."""
    length=np.linalg.norm(center)
    if length<1e-12 or not abs(length-2)<=radius<=length+2:
        return None
    axis=center/length
    height=(radius**2+length**2-4)/(2*length)
    rho=np.sqrt(max(0,radius**2-height**2))
    helper=np.eye(3)[np.argmin(abs(axis))]
    u=np.cross(axis,helper);u/=np.linalg.norm(u);v=np.cross(axis,u)
    angle=rng.uniform(0,2*np.pi,count)
    return height*axis+rho*(np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*v)


def merge(a,b,df,kf,rng,max_rotations=20):
    n1,n2=len(a),len(b);n=n1+n2
    target=(n/kf)**(2/df)
    distance2=n*n/(n1*n2)*(target-n1/n*rg2(a)-n2/n*rg2(b))
    if distance2<=0:raise RuntimeError('Infeasible mass-radius separation')
    distance=np.sqrt(distance2)
    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
    for _ in range(max_rotations):
        aa=a@rotation(rng).T;bb=b@rotation(rng).T
        centers=(aa[:,None]-bb[None,:]).reshape(-1,3)
        tree=cKDTree(centers)
        norms=np.linalg.norm(centers,axis=1)
        feasible=np.flatnonzero((abs(norms-2)<=distance)&(distance<=norms+2)&(norms>1e-12))
        for idx in rng.permutation(feasible)[:120]:
            candidates=circle_points(centers[idx],distance,rng)
            gaps,_=tree.query(candidates,k=1)
            for t,gap in zip(candidates,gaps):
                if gap >= 2-2e-11:
                    p=np.vstack([aa,bb+t]);p-=p.mean(axis=0)
                    # Strong independent contact/connectivity and overlap check.
                    inspect(p)
                    if not np.isclose(rg2(p),target,rtol=1e-10):raise AssertionError('Mass-radius mismatch')
                    return p
    raise RuntimeError(f'No non-overlapping merge for N={n}, Df={df}, kf={kf}')


def generate(n,df,kf,rng):
    if n<25 or n%25:raise ValueError('Particle count must be a positive multiple of 25')
    if n==25:
        for attempt in range(30):
            p=np.array([[-1.,0,0],[1.,0,0]])
            try:
                for _ in range(2,25):p=merge(p,np.zeros((1,3)),df,kf,rng,max_rotations=1)
                return p
            except RuntimeError:continue
        raise RuntimeError('25-particle seed construction exhausted')
    for attempt in range(20):
        left=25*((n//25)//2)
        a=generate(left,df,kf,rng);b=generate(n-left,df,kf,rng)
        try:return merge(a,b,df,kf,rng)
        except RuntimeError:continue
    raise RuntimeError('Hierarchical cluster construction exhausted')
