"""Check morphology coverage and group separation for every planned experiment."""
from train import splits
records=[]
for n in (50,100,200,500):
    for df,kf in ((1.8,1.3),(2.2,1.1),(2.6,.8)):
        for rep in range(15):
            records.append(dict(method='tunable_fractal',n=n,requested_df=df,requested_kf=kf,group=f'tunable_fractal|N{n}|Df{df}|kf{kf}'))
for subset,mode in [(records,'grouped'),([g for g in records if g['n']!=500],'grouped'),(records,'size500')]:
    for train,test,fit,val in splits(subset,mode):
        assert {subset[i]['requested_df'] for i in val} == {1.8,2.2,2.6}
        assert {subset[i]['requested_df'] for i in fit} == {1.8,2.2,2.6}
        assert set(fit)|set(val)==set(train)
        for left,right in ((fit,val),(train,test)):
            assert not {subset[i]['group'] for i in left}&{subset[i]['group'] for i in right}
        if mode=='size500':
            assert all(subset[i]['n']==500 for i in test)
            assert all(subset[i]['n']!=500 for i in train)
print('All core, extended and size-extrapolation splits passed.')

# A single missing realization must not destroy morphology coverage.
reduced=records[:20]+records[21:]
for subset in (reduced,[g for g in reduced if g["n"]!=500]):
    for train,test,fit,val in splits(subset):
        assert {subset[i]["requested_df"] for i in fit} == {1.8,2.2,2.6}
        assert {subset[i]["requested_df"] for i in val} == {1.8,2.2,2.6}
print("Single-exclusion splits passed.")
