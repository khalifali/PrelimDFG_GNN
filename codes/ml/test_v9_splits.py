"""Verify exact reference splits and non-overlapping test cases."""
from types import SimpleNamespace
import numpy as np
from train import splits
from reference_v9.train_gnn_pure_graph_latent_v9 import make_grouped_folds,realization_group
for sizes in ((50,100,200),(50,100,200,500)):
    records=sorted([dict(case=f'fractal_dp{d:g}_N{n:04d}_Df{df:g}_kf{kf:g}_rep{rep:02d}')
        for d in (1.,1.5,2.) for n in sizes for df,kf in ((1.8,1.3),(2.2,1.1),(2.6,.8)) for rep in range(1,6)
        if not (d==2 and n==100 and df==2.6 and rep==4)],key=lambda g:g['case'])
    reference=make_grouped_folds([SimpleNamespace(case=g['case']) for g in records],5,7)
    parts=splits(records,'v9');seen=[]
    for fold,(train,test,fit,val) in enumerate(parts):
        np.testing.assert_array_equal(test,reference[fold]);seen.extend(test)
        shuffled=np.random.default_rng(7+fold).permutation(np.setdiff1d(np.arange(len(records)),test))
        nval=max(1,int(round(.15*len(shuffled))))
        np.testing.assert_array_equal(val,shuffled[:nval]);np.testing.assert_array_equal(fit,shuffled[nval:])
        assert not {realization_group(records[i]['case']) for i in train}&{realization_group(records[i]['case']) for i in test}
    assert sorted(seen)==list(range(len(records)))
print('V9 core and extended partitions match the uploaded reference exactly.')
