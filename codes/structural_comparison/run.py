"""Compare graph and structural baselines on frozen pair-preserving development folds."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'structure'),str(HERE.parent/'ml'),str(HERE.parent/'matched_structure')]
from run import balanced_folds,ann_fit
from analyze import BASIC,EXTRA,write_csv
from train import train_neural,scores
from models import predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import torch

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    expected=json.loads((a.matched/'summary.json').read_text())['source_sha256']
    assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected, 'Source artifact changed'
    records=json.loads(a.source.read_text())['records']
    rows=list(csv.DictReader((a.matched/'structural_features.csv').open()))
    pairs=list(csv.DictReader((a.matched/'pairs_primary.csv').open()))
    by={r['case']:r for r in rows}
    assert len(records)==359 and len(pairs)==167
    parents={g['independence_group']:g['independence_group'] for g in records}
    groups={g['case']:g['independence_group'] for g in records}
    def root(k):
        while parents[k]!=k:k=parents[k]
        return k
    for p in pairs:parents[root(groups[p['a']])]=root(groups[p['b']])
    for g in records:
        g['independence_group']=root(g['independence_group'])
        for k in ('pos','nodes','descriptors','edge'):
            g[k]=np.asarray(g[k],dtype=np.int64 if k=='edge' else np.float32)
        r=by[g['case']]
        assert abs(float(r['target'])-g['target'])<1e-12 and int(r['n'])==g['n']
        g['basic']=np.array([np.log(g['n'])]+[float(r[c]) for c in BASIC])
        g['expanded']=np.r_[g['basic'],[float(r[c]) for c in EXTRA]]
    labels=balanced_folds(records,9192026)
    ids=np.arange(len(records));splits=[];frozen=[]
    for fold in range(5):
        outer=ids[labels!=fold];test=ids[labels==fold]
        inner=balanced_folds([records[i] for i in outer],9192100+fold)
        fit=outer[inner!=0];val=outer[inner==0]
        sets=[{records[i]['independence_group'] for i in ii} for ii in (fit,val,test)]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        splits.append((fit,val,test))
        for role,ii in [('fit',fit),('validation',val),('test',test)]:
            frozen.extend(dict(fold=fold,role=role,case=records[i]['case'],group=records[i]['independence_group']) for i in ii)
    write_csv(a.output/'splits.csv',frozen)
    provenance=dict(status='development only; previously inspected geometries',seed=a.seed,
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        features=dict(basic=['log_n']+BASIC,expanded=['log_n']+BASIC+EXTRA),
        selection='minimum validation absolute-exposure MSE; q ties within numerical precision choose smallest',
        pair_evaluation='difference of two held-out absolute exposure predictions',
        split_sha256=hashlib.sha256((a.output/'splits.csv').read_bytes()).hexdigest())
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if a.verify_only:
        print('Verified 359 cases, 167 intact pairs and all five fit/validation/test partitions',flush=True)
        return
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    y=np.array([g['target'] for g in records]);predictions=[];selections=[]
    for fold,(fit,val,test) in enumerate(splits):
        def record(name,p):
            assert np.isfinite(p).all()
            predictions.extend(dict(fold=fold,seed=a.seed,model=name,case=records[i]['case'],target=float(y[i]),prediction=float(v)) for i,v in zip(test,p))
            write_csv(a.output/'predictions.csv',predictions)
            print(dict(fold=fold,seed=a.seed,model=name,**scores(y[test],p)),flush=True)
        for feature in ('basic','expanded'):
            x=np.array([g[feature] for g in records])
            for kind in ('ridge','forest'):
                candidates=[]
                for param in ([.01,.1,1,10,100,1000] if kind=='ridge' else [1,3,5]):
                    model=(make_pipeline(StandardScaler(),Ridge(alpha=param)) if kind=='ridge' else
                        RandomForestRegressor(n_estimators=500,min_samples_leaf=param,random_state=a.seed,n_jobs=2))
                    model.fit(x[fit],y[fit])
                    candidates.append((float(np.mean((model.predict(x[val])-y[val])**2)),param,model))
                loss,param,model=min(candidates,key=lambda z:z[0])
                record(kind+'_'+feature,model.predict(x[test]))
                selections.append(dict(fold=fold,model=kind+'_'+feature,parameter=param,validation_mse=loss))
            pp,sel=ann_fit(x[fit],y[fit],x[val],y[val],a.seed,epochs=a.epochs)
            record('ann_'+feature,pp(x[test]))
            selections.append(dict(fold=fold,model='ann_'+feature,parameter=sel['epoch'],validation_mse=sel['validation_mse']))
        candidates=[]
        for q in (1,2,3,4):
            model,scale,epochs,_=train_neural([records[i] for i in fit],[records[i] for i in val],'gnn',q,a.seed+1000*q+fold,a.epochs,50)
            loss=float(np.mean((predict(model,[records[i] for i in val],scale,'gnn')[0]-y[val])**2))
            p=predict(model,[records[i] for i in test],scale,'gnn')[0]
            record('gnn_q'+str(q),p);candidates.append((loss,q,p))
            selections.append(dict(fold=fold,model='gnn_q'+str(q),parameter=epochs,validation_mse=loss))
        loss,q,p=min(candidates,key=lambda z:(z[0],z[1]))
        record('gnn_selected',p)
        selections.append(dict(fold=fold,model='gnn_selected',parameter=q,validation_mse=loss))
        write_csv(a.output/'selections.csv',selections)
    metrics=[];pair_predictions=[]
    for name in sorted({r['model'] for r in predictions}):
        rr=[r for r in predictions if r['model']==name];lookup={r['case']:r for r in rr}
        assert len(lookup)==359
        metrics.append(dict(model=name,scope='absolute_exposure',seed=a.seed,n=len(rr),**scores([r['target'] for r in rr],[r['prediction'] for r in rr])))
        pp=[]
        for pair in pairs:
            x,z=lookup[pair['a']],lookup[pair['b']];assert x['fold']==z['fold']
            pp.append(dict(model=name,seed=a.seed,a=pair['a'],b=pair['b'],fold=x['fold'],target=z['target']-x['target'],prediction=z['prediction']-x['prediction']))
        pair_predictions+=pp
        metrics.append(dict(model=name,scope='matched_pair_difference',seed=a.seed,n=len(pp),**scores([r['target'] for r in pp],[r['prediction'] for r in pp])))
    write_csv(a.output/'metrics.csv',metrics);write_csv(a.output/'pair_predictions.csv',pair_predictions)
    lines=['# Structural baseline versus GNN — development comparison','',
        'All 359 cases are previously inspected development data. Primary matched pairs stay together in fit, validation and test. No requested Df/kf inputs. All models use identical cases and absolute-exposure validation MSE for selection. No outer refit. Fixed q results are sensitivity analyses, not test-based choices. Pair differences are secondary evaluations, not the training objective.','',
        '| Model | Scope | R² | RMSE | MAE |','|---|---|---:|---:|---:|']
    for r in metrics:lines.append(f"| {r['model']} | {r['scope']} | {r['r2']:.5f} | {r['rmse']:.6f} | {r['mae']:.6f} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--matched',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=300);p.add_argument('--verify-only',action='store_true');main(p.parse_args())
