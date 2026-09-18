"""Audit measured morphology and compare models on geometry-defined development splits."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'ml'),str(HERE.parent/'postprocessing')]
from descriptors import measure
from dataset import read_dataset,DESCRIPTORS
from posttovtk import frames

FEATURES=DESCRIPTORS+['box_dimension','hull_porosity']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_positions(p):
    ff=list(frames(p))
    if len(ff)!=1:raise ValueError('One snapshot required')
    step,_,columns,rows,_=ff[0];arr=np.asarray(rows,float)
    arr=arr[np.argsort(arr[:,columns.index('id')])]
    return arr[:,[columns.index(k) for k in ('x','y','z')]],arr[:,columns.index('radius')]


def collect(a):
    from scipy.spatial.distance import pdist
    a.output.mkdir(parents=True,exist_ok=False)
    audits=[];curves=[];records=[]
    for label,dataset,campaign in [('original',a.original/'final_dataset.json',a.original/'bonded'),
        ('previous_independent',a.independent/'experiment/test_dataset.json',a.independent/'experiment/bonded')]:
        data=read_dataset(dataset)
        if len(data)!=(179 if label=='original' else 180):raise ValueError('Unexpected coverage')
        for g in data:
            folder=campaign/g['case'];meta=json.loads((folder/'case.json').read_text())
            assessment=json.loads((folder/'assessment.json').read_text())
            step=meta['parameters_SI']['steps']
            if digest(folder/f'post/post_{step}.txt')!=g['dump_sha256']:raise ValueError('Snapshot/target provenance mismatch')
            initial,r=read_positions(folder/'post/post_0.txt')
            key=hashlib.sha256(np.round(np.sort(pdist(initial/(2*r[0]))),7).astype('<f8').tobytes()).hexdigest()
            # Explicit lineage can be supplied by an importer; geometry fingerprint
            # additionally merges scaled/rotated copies with different filenames.
            lineage=meta['source_geometry'].get('root_geometry_id',key)
            g.update(independence_group=str(lineage),geometry_fingerprint=key,source_cohort=label)
            measurements={}
            for stage,filename in [('initial','post_0.txt'),('final',f'post_{step}.txt')]:
                xyz,radii=read_positions(folder/'post'/filename)
                values,bc=measure(xyz,radii);measurements[stage]=values
                audits.append(dict(case=g['case'],stage=stage,n=g['n'],source_cohort=label,
                    requested_df=g.get('requested_df',''),requested_kf=g.get('requested_kf',''),
                    mechanical_passed=assessment['passed'],**values))
                curves.extend(dict(case=g['case'],stage=stage,**c) for c in bc)
            g['measured']=measurements['final']
            g['expanded_descriptors']=np.r_[g['descriptors'],measurements['final']['box_dimension'],measurements['final']['hull_porosity']].tolist()
            g['descriptor_change']={k:measurements['final'][k]-measurements['initial'][k] for k in ('box_dimension','hull_porosity')}
            records.append(g)
            write_csv(a.output/'descriptors.csv',audits)
            print(f'Measured {len(records)}/359: {g["case"]}',flush=True)
    # Union provenance lineage and coincident-geometry groups, not measured-D bins.
    parents={g['case']:g['case'] for g in records}
    def find(k):
        while parents[k]!=k:parents[k]=parents[parents[k]];k=parents[k]
        return k
    seen={}
    for g in records:
        for kind in ('independence_group','geometry_fingerprint'):
            k=(kind,g[k])
            if k in seen:parents[find(g['case'])]=find(seen[k])
            else:seen[k]=g['case']
    for g in records:g['independence_group']=find(g['case'])
    write_csv(a.output/'box_counting_curves.csv',curves)
    serial=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in g.items()} for g in records]
    (a.output/'graphs.json').write_text(json.dumps(dict(features=FEATURES,records=serial)))
    flags=[r for r in audits if r['stage']=='final' and r['box_quality']!='ok']
    summary=dict(cases=len(records),independence_groups=len(set(g['independence_group'] for g in records)),
        final_box_quality_flag_count=len(flags),all_cases_retained=True,
        max_porosity_overlap_error_bound=max(r['porosity_overlap_error_bound'] for r in audits),
        max_absolute_box_change=max(abs(g['descriptor_change']['box_dimension']) for g in records),
        max_absolute_porosity_change=max(abs(g['descriptor_change']['hull_porosity']) for g in records),
        status='development only; previous tests have been inspected',
        sources=dict(original_dataset_sha256=digest(a.original/'final_dataset.json'),
                     previous_test_dataset_sha256=digest(a.independent/'experiment/test_dataset.json')))
    (a.output/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'source_hashes.json').write_text(json.dumps({p.name:digest(p) for p in HERE.glob('*.py')},indent=2))
    audit_report(a.output,audits,summary)
    print(json.dumps(summary,indent=2),flush=True)


def audit_report(output,audits,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final=[r for r in audits if r['stage']=='final']
    before={r['case']:r for r in audits if r['stage']=='initial'}
    lines=['# Measured structure audit','',
        'All measurements are geometry-only. Requested Df/kf are shown as provenance, not used as predictor inputs or fold strata.',
        '',f"Cases: {summary['cases']}; final box-dimension quality flags: {summary['final_box_quality_flag_count']}. No quality-flagged cases were removed.",
        '', '| Particle count | Cases | Box quality flags | Mean measured dimension | Mean hull porosity |', '|---|---:|---:|---:|---:|']
    for n in sorted({r['n'] for r in final}):
        rr=[r for r in final if r['n']==n]
        lines.append(f"| {n} | {len(rr)} | {sum(r['box_quality']!='ok' for r in rr)} | {np.mean([r['box_dimension'] for r in rr]):.4f} | {np.mean([r['hull_porosity'] for r in rr]):.4f} |")
    lines += ['', 'A high log-log fit R² over a short interval is not evidence of a well-resolved fractal regime. '
        'Reported slope standard errors are regression diagnostics, not uncertainty in an underlying true dimension. '
        'Grid spread and alternate-window sensitivity are recorded separately.', '',
        f"Maximum porosity error bound from neglecting sphere overlaps: {summary['max_porosity_overlap_error_bound']:.6g}.",
        '', 'Convex-hull porosity includes empty space between branches; it is not local pore-network porosity.',
        '', '![Measured structural coverage](audit.png)']
    (output/'audit.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    colors=np.log10([r['n'] for r in final])
    axes[0,0].scatter([r['requested_df'] for r in final],[r['box_dimension'] for r in final],c=colors,s=14)
    axes[0,0].set(xlabel='Requested mass–radius exponent',ylabel='Measured box-counting exponent')
    sc=axes[0,1].scatter([r['hull_porosity'] for r in final],[r['box_dimension'] for r in final],c=colors,s=14)
    axes[0,1].set(xlabel='Particle-body hull porosity',ylabel='Measured box-counting exponent')
    fig.colorbar(sc,ax=axes[0,:],label='log10(particle count)')
    axes[1,0].scatter([r['n'] for r in final],[r['box_scale_span_decades'] for r in final],s=14)
    axes[1,0].axhline(.5,color='red',linestyle='--',linewidth=1)
    axes[1,0].set(xlabel='Particle count',ylabel='Fitted scale span (decades)',xscale='log')
    axes[1,1].scatter([r['n'] for r in final],[r['box_dimension']-before[r['case']]['box_dimension'] for r in final],s=14)
    axes[1,1].set(xlabel='Particle count',ylabel='Post-BPM minus initial exponent',xscale='log')
    fig.savefig(output/'audit.png',dpi=180);plt.close(fig)


def group_values(records):
    groups=sorted(set(g['independence_group'] for g in records))
    values=[]
    for group in groups:
        rows=[g for g in records if g['independence_group']==group]
        values.append(np.mean([[np.log(g['n']),g['measured']['box_dimension'],g['measured']['hull_porosity']] for g in rows],axis=0))
    return groups,np.asarray(values)


def balanced_folds(records,seed=8131):
    """Balance measured morphology strata; never split provenance groups."""
    groups,values=group_values(records)
    bins=np.column_stack([np.searchsorted(np.quantile(values[:,k],[1/3,2/3]),values[:,k],side='right') for k in range(3)])
    rng=np.random.default_rng(seed);assignment={};totals=np.zeros(5,int)
    for cell in sorted(set(map(tuple,bins))):
        ids=np.flatnonzero(np.all(bins==cell,axis=1));rng.shuffle(ids)
        local=np.zeros(5,int)
        for i in ids:
            choices=np.flatnonzero(local==local.min());choices=choices[totals[choices]==totals[choices].min()]
            fold=int(rng.choice(choices));assignment[groups[i]]=fold;local[fold]+=1;totals[fold]+=1
    return np.array([assignment[g['independence_group']] for g in records])


def partitions(records):
    labels=balanced_folds(records);all_ids=np.arange(len(records));result=[]
    for fold in range(5):result.append((f'geometry_interpolation_fold{fold}',all_ids[labels!=fold],all_ids[labels==fold],'balanced'))
    groups,values=group_values(records)
    threshold=np.quantile(values[:,2],.8)
    held=set(g for g,v in zip(groups,values) if v[2]>=threshold)
    for label,held,mode in [('high_porosity_region',held,'porosity'),
        ('large_size_region',{g['independence_group'] for g in records if g['n']>=750},'size')]:
        mask=np.array([g['independence_group'] in held for g in records])
        result.append((label,all_ids[~mask],all_ids[mask],mode))
    output=[]
    for name,outer,test,mode in result:
        train=[records[i] for i in outer]
        if mode=='balanced':val_mask=balanced_folds(train,9173)==0
        else:
            groups,values=group_values(train);k=2 if mode=='porosity' else 0
            threshold=np.quantile(values[:,k],.8)
            held=set(g for g,v in zip(groups,values) if v[k]>=threshold)
            val_mask=np.array([g['independence_group'] in held for g in train])
        fit=outer[~val_mask];val=outer[val_mask]
        sets=[{records[i]['independence_group'] for i in idx} for idx in (fit,val,test)]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or min(map(len,sets))<2:
            raise ValueError('Invalid dependent-group split')
        output.append((name,fit,val,test))
    return output


def ann_fit(x,y,xv,yv,seed,epochs=300):
    import copy,torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(seed);scaler=StandardScaler().fit(x)
    mean=float(y.mean());scale=max(float(y.std(ddof=1)),1e-8)
    xx=torch.tensor(scaler.transform(x),dtype=torch.float32);yy=torch.tensor((y-mean)/scale,dtype=torch.float32)
    vv=torch.tensor(scaler.transform(xv),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    rng=np.random.default_rng(seed);best=float('inf');stale=0;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train()
        order=rng.permutation(len(x))
        for start in range(0,len(order),16):
            idx=order[start:start+16];opt.zero_grad();loss=((model(xx[idx]).flatten()-yy[idx])**2).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        model.eval()
        with torch.no_grad():pred=model(vv).flatten().numpy()*scale+mean
        error=float(np.mean((pred-yv)**2))
        if error<best:best=error;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch
        else:stale+=1
        if stale>=50:break
    model.load_state_dict(state);model.eval()
    def predict(a):
        with torch.no_grad():return model(torch.tensor(scaler.transform(a),dtype=torch.float32)).flatten().numpy()*scale+mean
    return predict,dict(epoch=best_epoch,validation_mse=best)


def train(a):
    import torch
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from train import train_neural,scores
    from models import predict
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    records=json.loads((a.output/'graphs.json').read_text())['records']
    for g in records:
        for k in ('pos','nodes','descriptors','edge'):g[k]=np.array(g[k],dtype=np.int64 if k=='edge' else np.float32)
    splits=partitions(records);frozen=[]
    for name,fit,val,test in splits:
        for role,indices in [('fit',fit),('validation',val),('test',test)]:
            frozen.extend(dict(protocol=name,role=role,case=records[i]['case'],group=records[i]['independence_group']) for i in indices)
    write_csv(a.output/'splits.csv',frozen)
    fold_summary=[]
    for name,fit,val,test in splits:
        for role,indices in [('fit',fit),('validation',val),('test',test)]:
            rr=[records[i] for i in indices]
            fold_summary.append(dict(protocol=name,role=role,cases=len(rr),
                independent_groups=len({r['independence_group'] for r in rr}),
                min_N=min(r['n'] for r in rr),max_N=max(r['n'] for r in rr),
                min_box_dimension=min(r['measured']['box_dimension'] for r in rr),
                max_box_dimension=max(r['measured']['box_dimension'] for r in rr),
                min_porosity=min(r['measured']['hull_porosity'] for r in rr),
                max_porosity=max(r['measured']['hull_porosity'] for r in rr)))
    write_csv(a.output/'fold_summary.csv',fold_summary)
    metrics=[];rows=[];selections=[]
    for fold,(name,fit,val,test) in enumerate(splits):
        subset=lambda ids:[records[i] for i in ids]
        y=np.array([g['target'] for g in records]);yy=y[test]
        for seed in (7,17,27):
            def record(model,p):
                if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
                metrics.append(dict(protocol=name,seed=seed,model=model,n=len(test),**scores(yy,p)))
                rows.extend(dict(protocol=name,seed=seed,model=model,case=records[i]['case'],target=float(y[i]),prediction=float(v)) for i,v in zip(test,p))
                write_csv(a.output/'metrics.csv',metrics);write_csv(a.output/'predictions.csv',rows)
                print(metrics[-1],flush=True)
            for features,key in [('original6','descriptors'),('expanded8','expanded_descriptors')]:
                x=np.array([g[key] for g in records])
                linear=make_pipeline(StandardScaler(),LinearRegression()).fit(x[fit],y[fit])
                record('linear_'+features,linear.predict(x[test]))
                candidates=[]
                for leaf in (1,3,5):
                    rf=RandomForestRegressor(n_estimators=500,min_samples_leaf=leaf,max_features=1.,random_state=seed,n_jobs=2).fit(x[fit],y[fit])
                    candidates.append((np.mean((rf.predict(x[val])-y[val])**2),leaf,rf))
                error,leaf,rf=min(candidates,key=lambda c:c[0]);record('forest_'+features,rf.predict(x[test]))
                selections.append(dict(protocol=name,seed=seed,model='forest_'+features,selection=json.dumps(dict(leaf=leaf,validation_mse=float(error)))))
                predictor,selection=ann_fit(x[fit],y[fit],x[val],y[val],seed)
                record('ann_'+features,predictor(x[test]))
                selections.append(dict(protocol=name,seed=seed,model='ann_'+features,selection=json.dumps(selection)))
            neural={}
            for q in (1,2,3,4):
                model,scale,epochs,_=train_neural(subset(fit),subset(val),'gnn',q,seed+1000*q+fold,300,50)
                validation=predict(model,subset(val),scale,'gnn')[0]
                pp=predict(model,subset(test),scale,'gnn')[0]
                neural[q]=(scores(y[val],validation)['r2'],pp)
                record(f'gnn_q{q}',pp)
                selections.append(dict(protocol=name,seed=seed,model=f'gnn_q{q}',selection=json.dumps(dict(epoch=epochs,validation_r2=neural[q][0]))))
            best=max(v[0] for v in neural.values());q=min(q for q in neural if neural[q][0]>=best-.02)
            record('gnn_selected',neural[q][1])
            selections.append(dict(protocol=name,seed=seed,model='gnn_selected',selection=json.dumps(dict(q=q))))
            write_csv(a.output/'selections.csv',selections)
    report(a.output,rows)


def report(output,rows):
    from train import scores
    lines=['# Measured-geometry development comparison','','All 359 previously inspected cases are development data. '
        'These scores are not fresh independent confirmation. All models use identical fit/validation/test cases. '
        'GNN receives geometry only; expanded baselines add measured box dimension and particle-body hull porosity. '
        'Requested Df/kf do not determine these folds. One fixed-time kinetic-energy exception remains included.','',
        'Mean ± sample SD across three training seeds is not a confidence interval. Box quality flags are retained in descriptors.csv.','']
    for protocol in ('geometry_interpolation','high_porosity_region','large_size_region'):
        lines += [f'## {protocol}','','| Model | R² | RMSE | MAE |','|---|---:|---:|---:|']
        for model in sorted({r['model'] for r in rows}):
            values=[]
            for seed in (7,17,27):
                rr=[r for r in rows if r['protocol'].startswith(protocol) and r['seed']==seed and r['model']==model]
                values.append(scores([r['target'] for r in rr],[r['prediction'] for r in rr]))
            cells=[f"{np.mean([v[k] for v in values]):.5f} ± {np.std([v[k] for v in values],ddof=1):.5f}" for k in ('r2','rmse','mae')]
            lines.append('| '+model+' | '+' | '.join(cells)+' |')
        lines.append('')
    (output/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['collect','train'])
    p.add_argument('--original',type=Path);p.add_argument('--independent',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)
