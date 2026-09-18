"""Independent numerical and split-isolation checks."""
import ast
import itertools
from pathlib import Path
import numpy as np
from descriptors import body_hull_volume,count_boxes,measure
from run import partitions,balanced_folds


def main():
    cube=np.array(list(itertools.product((-1.,1.),repeat=3)))
    r=.3
    exact=8+24*r+6*np.pi*r*r+4*np.pi*r**3/3
    assert np.isclose(body_hull_volume(cube,r),exact,rtol=1e-12)
    assert np.isclose(body_hull_volume(np.zeros((1,3)),r),4*np.pi*r**3/3)
    line=np.array([[0.,0,0],[5,0,0]])
    assert np.isclose(body_hull_volume(line,r),5*np.pi*r*r+4*np.pi*r**3/3)
    square=np.array([[0.,0,0],[2,0,0],[2,2,0],[0,2,0]])
    assert np.isclose(body_hull_volume(square,r),8*r+4*np.pi*r*r+4*np.pi*r**3/3)
    # Compare occupied-box counts with the actual legacy intersection/count code
    # on a common fully containing domain. No GUI imports are needed.
    legacy=Path(__file__).parents[1]/'fractalagglomerategenration/DimensionAnalysis/testdim.py'
    tree=ast.parse(legacy.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sphere_box_intersect','count_occupied_boxes')]
    namespace={'np':np};exec(compile(tree,str(legacy),'exec'),namespace)
    pp=np.array([[.12,.23,.31],[1.17,2.24,1.45],[3.12,2.52,3.36]])
    origin=np.array([-2.,-2.,-2.]);L=12.
    for n in (3,6,12):
        old=namespace['count_occupied_boxes'](pp,np.full(3,.5),origin,L,n)
        assert old==count_boxes(pp,.5,L/n,origin)
    cloud=np.array(list(itertools.product(np.arange(5)*1.1,repeat=3)))
    first,_=measure(cloud,np.full(len(cloud),.5))
    second,_=measure(cloud*3+7,np.full(len(cloud),1.5))
    assert np.isclose(first['hull_porosity'],second['hull_porosity'],atol=1e-12)
    assert np.isclose(first['box_dimension'],second['box_dimension'],atol=1e-8)
    rows=[]
    for i in range(80):
        for copy in range(2):
            rows.append(dict(case=f'{i}_{copy}',independence_group=str(i),n=[50,100,200,500,750,1000][i%6],
                measured=dict(box_dimension=1.5+(i%11)/10,hull_porosity=.2+(i%17)/25)))
    folds=balanced_folds(rows)
    assert all(folds[2*i]==folds[2*i+1] for i in range(80))
    assert len(partitions(rows))==7
    # Requested generator settings and targets cannot influence the split.
    changed=[dict(g,requested_df=99,requested_kf=-1,target=123) for g in rows]
    assert np.array_equal(folds,balanced_folds(changed))
    print('Hull formulas, legacy box counts, scale invariance and group isolation passed.')


if __name__=='__main__':main()
