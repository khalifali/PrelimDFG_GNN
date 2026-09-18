"""Checks for held-out design and development-only selection."""
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from campaign import HERE, planned_cases, freeze, signature


def main():
    p = json.loads((HERE/'protocol.json').read_text())
    cases = list(planned_cases(p))
    assert len(cases) == len({r['case'] for r in cases}) == len({r['seed'] for r in cases}) == 180
    assert sum(r['subset']=='interpolation' for r in cases) == 90
    assert sum(r['subset']=='extrapolation' for r in cases) == 90
    for r in cases:
        if r['subset']=='interpolation':
            assert 50 < r['n'] < 500 and r['n'] not in (50,100,200,500)
            assert 1.8 < r['requested_df'] < 2.6
        else:
            assert r['n'] > 500
    points = np.array([[0.,0,0],[2.,0,0],[2.,2.,0],[0.,0,2.]])
    q, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(3,3)))
    assert signature(points) == signature((points@q+4)[::-1])
    changed = points.copy(); changed[-1,-1] += .01
    assert signature(points) != signature(changed)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp); data=root/'dev.json'; selections=root/'selections.csv'
        data.write_text(json.dumps(dict(contact_tolerance_over_d=1e-6, records=[
            dict(case=f'dev{i}', snapshot='final', rays=2048) for i in range(179)])))
        rows=[]
        for fold in range(5):
            for seed in (7,17,27):
                for kind in ('gnn','ann'):
                    for q in (1,2,3,4):
                        # q=2 qualifies under the .02 tolerance; q=1 does not.
                        rows.append(dict(fold=fold, seed=seed, model=f'{kind}_q{q}',
                            selection=json.dumps(dict(inner_r2={1:.80,2:.94,3:.95,4:.945}[q], best_epoch=20+fold))))
                rows.append(dict(fold=fold,seed=seed,model='ann',selection=json.dumps(dict(best_epoch=25))))
                rows.append(dict(fold=fold,seed=seed,model='random_forest',selection=json.dumps(dict(min_samples_leaf=3,max_features=1.))))
        with selections.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['fold','seed','model','selection']);w.writeheader();w.writerows(rows)
        a=SimpleNamespace(dataset=data,selections=selections,output=root/'experiment')
        freeze(a)
        frozen=json.loads((a.output/'freeze.json').read_text())
        assert frozen['selected_q']==dict(gnn=2,ann=2)
        assert frozen['epochs']['gnn_q2']==22
        assert frozen['forest']==dict(min_samples_leaf=3,max_features=1.)
        assert len(frozen['planned_cases'])==180
        # Refuse to overwrite an existing registered experiment.
        try:
            freeze(a)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing freeze was overwritten')
    print('Independent design, selection isolation, duplicate detection and freeze protection passed.')


if __name__=='__main__':
    main()
