"""Grouped outer evaluation; all scaling, selection and stopping use training data."""
import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import platform
from pathlib import Path
import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from dataset import read_dataset, DESCRIPTORS
from models import make_model, scaler, inputs, predict


def scores(y, p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y,p))), mae=float(mean_absolute_error(y,p)), r2=float(r2_score(y,p)))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def v9_partitions(records):
    # Use the uploaded source itself, avoiding a second implementation of its
    # diameter/N/Df grouping and size-balanced assignment. Hold split seed fixed
    # across neural seeds so variability reflects training, not changed tests.
    from types import SimpleNamespace
    from reference_v9.train_gnn_pure_graph_latent_v9 import make_grouped_folds
    data = [SimpleNamespace(case=g['case']) for g in records]
    folds = make_grouped_folds(data, 5, 7)
    order = np.arange(len(records))
    result = []
    for fold, test in enumerate(folds):
        train = np.setdiff1d(order, test)
        shuffled = np.random.default_rng(7 + fold).permutation(train)
        nval = max(1, int(round(.15 * len(shuffled))))
        val, fit = shuffled[:nval], shuffled[nval:]
        assert not set(train) & set(test)
        assert not set(fit) & set(val)
        result.append((train, test, fit, val))
    return result


def splits(records, mode='grouped'):
    if mode == 'v9':
        return v9_partitions(records)
    groups = np.array([g['group'] for g in records])
    all_ids = np.arange(len(records))
    if mode == 'grouped':
        if all(g['method'] == 'tunable_fractal' for g in records):
            # Balance parameter groups across morphologies, independent of targets.
            # GroupKFold's sample-count ordering changes after one exclusion and
            # can otherwise put two of the three core groups of a morphology
            # into the same test fold, leaving no separate inner-validation group.
            by_morphology = {}
            for g in records:
                key = (g['requested_df'], g['requested_kf'])
                by_morphology.setdefault(key, set()).add(g['group'])
            assignment = {}
            cursor = 0
            rng = np.random.default_rng(701)
            for key in sorted(by_morphology):
                names = sorted(by_morphology[key])
                rng.shuffle(names)
                for name in names:
                    assignment[name] = cursor % 5
                    cursor += 1
            labels = np.array([assignment[g] for g in groups])
            outer = [(all_ids[labels != f], all_ids[labels == f]) for f in range(5)]
        else:
            outer = list(GroupKFold(n_splits=5).split(all_ids, groups=groups))
    elif mode == 'method':
        methods = np.array([g['method'] for g in records])
        outer = [(all_ids[methods != m], all_ids[methods == m]) for m in sorted(set(methods))]
    else:
        outer = [(np.array([i for i,g in enumerate(records) if g['n'] != 500]), np.array([i for i,g in enumerate(records) if g['n'] == 500]))]
    result = []
    for fold,(train,test) in enumerate(outer):
        if all(g['method'] == 'tunable_fractal' for g in records):
            # Keep morphology coverage in validation without separating any
            # N/Df/kf group or consulting exposure targets or test performance.
            rng = np.random.default_rng(701 + fold)
            by_morphology = {}
            for i in train:
                key = (records[i]['requested_df'], records[i]['requested_kf'])
                by_morphology.setdefault(key, set()).add(groups[i])
            validation_groups = set()
            for key in sorted(by_morphology):
                choices = sorted(by_morphology[key])
                if len(choices) < 2:
                    raise ValueError('Need at least two training groups per morphology')
                validation_groups.add(choices[int(rng.integers(len(choices)))])
            val = np.array([i for i in train if groups[i] in validation_groups])
            fit = np.array([i for i in train if groups[i] not in validation_groups])
        else:
            inner_train, inner_val = next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=701+fold).split(train,groups=groups[train]))
            fit, val = train[inner_train], train[inner_val]
        assert not set(groups[train]) & set(groups[test])
        assert not set(groups[fit]) & set(groups[val])
        result.append((train,test,fit,val))
    return result


def train_neural(training, validation, kind, q, seed, epochs, patience, fixed_epochs=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = make_model(kind,q)
    scale = scaler(training)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    y = np.array([g['target'] for g in training])
    target = torch.tensor((y-scale['target_mean'])/scale['target_scale'], dtype=torch.float32)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    best_state, history = None, []
    rng = np.random.default_rng(seed)
    for epoch in range(1,(fixed_epochs or epochs)+1):
        model.train()
        order = rng.permutation(len(training))
        loss_total = 0.
        for start in range(0,len(order),16):
            idx = order[start:start+16]
            batch = [training[i] for i in idx]
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs(batch,scale,kind,rotate=kind=='gnn'))
            loss = torch.mean((output-target[idx])**2)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            loss_total += loss.item()*len(idx)
        row = dict(epoch=epoch, training_rmse=float(np.sqrt(loss_total/len(training))*scale['target_scale']))
        if validation is not None:
            val_pred,_ = predict(model,validation,scale,kind)
            val_loss = mean_squared_error([g['target'] for g in validation],val_pred)
            row['validation_rmse'] = float(np.sqrt(val_loss))
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(row)
        if fixed_epochs is None and stale >= patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
    return model, scale, best_epoch, history


def save_checkpoint(path, model, scale, kind, q, epochs, cases):
    torch.save(dict(schema=1,kind=kind,q=q,state_dict=model.state_dict(),
                    scale={k:np.asarray(v).tolist() for k,v in scale.items()},epochs=epochs,
                    training_cases=[g['case'] for g in cases],descriptors=DESCRIPTORS),path)


def run(a):
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    records = read_dataset(a.dataset)
    if a.particle_counts:
        records = [g for g in records if g['n'] in a.particle_counts]
    if a.mode == 'v9':
        from reference_v9.train_gnn_pure_graph_latent_v9 import realization_group
        records = sorted(records, key=lambda g: g['case'])
        for g in records:
            g['group'] = realization_group(g['case'])
    if not records or len({g['group'] for g in records}) < 5:
        raise ValueError('At least five independent groups are required')
    out = a.output
    out.mkdir(parents=True,exist_ok=True)
    previous = json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else None
    if previous and not a.resume:
        raise ValueError('Output already contains a run; choose a fresh directory or --resume')
    digest = hashlib.sha256(a.dataset.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        commit = 'unknown'
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    provenance = dict(code_sha256=source_hashes,dataset_sha256=digest,dataset=str(a.dataset),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                      versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),commit=commit,
                      target='mean surface-normal geometric exposure',status='running',smoke=a.epochs<50,
                      cases=[g['case'] for g in records])
    if previous:
        ignored={'resume','dataset','output','verify_splits'}
        old={k:v for k,v in previous['arguments'].items() if k not in ignored}
        new={k:v for k,v in provenance['arguments'].items() if k not in ignored}
        if previous['dataset_sha256'] != digest or previous['code_sha256'] != source_hashes or old != new:
            raise ValueError('Resume requires identical data, code and training settings')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    partitions = splits(records,a.mode)
    frozen = []
    for fold,(train,test,fit,val) in enumerate(partitions):
        for role,idx in [('train',fit),('validation',val),('test',test)]:
            frozen.extend(dict(fold=fold,case=records[i]['case'],group=records[i]['group'],role=role) for i in idx)
    write_csv(out/'splits.csv',frozen)
    if a.verify_splits and a.verify_splits.read_text() != (out/'splits.csv').read_text():
        raise ValueError('Paired dataset splits differ')
    rows, selections, metrics = [], [], []
    completed = {}
    if previous:
        for file in out.glob('fold*/seed*/complete.json'):
            saved=json.loads(file.read_text())
            completed[(saved['fold'],saved['seed'])]=saved
            rows.extend(saved['predictions']);metrics.extend(saved['metrics']);selections.extend(saved['selections'])
    for fold,(train,test,fit,val) in enumerate(partitions):
        subset = lambda idx:[records[i] for i in idx]
        outer, testing, inner, validation = map(subset,(train,test,fit,val))
        x = lambda data:np.stack([g['descriptors'] for g in data])
        y = lambda data:np.array([g['target'] for g in data])
        for seed in a.seeds:
            if (fold,seed) in completed:
                print(f'Reusing completed fold={fold} seed={seed}',flush=True)
                continue
            folder = out/f'fold{fold}'/f'seed{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            def record(name,p,latent=None):
                if not np.isfinite(p).all():
                    raise ValueError('Non-finite predictions')
                metrics.append(dict(fold=fold,seed=seed,model=name,n=len(testing),**scores(y(testing),p)))
                for j,g in enumerate(testing):
                    rows.append(dict(case=g['case'],group=g['group'],method=g['method'],n=g['n'],fold=fold,seed=seed,model=name,target=g['target'],prediction=float(p[j])))
                write_csv(out/'predictions.csv',rows)
                write_csv(out/'fold_metrics.csv',metrics)
                print(f'fold={fold} seed={seed} {name}: {metrics[-1]}',flush=True)
            record('training_mean',np.full(len(testing),y(outer).mean()))
            linear = make_pipeline(StandardScaler(),LinearRegression()).fit(x(outer),y(outer))
            joblib.dump(linear,folder/'linear.joblib')
            record('linear',linear.predict(x(testing)))
            # Retain the uploaded v9 fixed forest and count/individual baselines.
            baseline_specs = [
                ('random_forest_v9', list(range(6)), RandomForestRegressor(n_estimators=a.trees,max_depth=4,min_samples_leaf=3,random_state=seed,n_jobs=a.threads)),
                ('count_quadratic',[0],make_pipeline(PolynomialFeatures(degree=2,include_bias=False),LinearRegression())),
                ('count_categorical',[0],make_pipeline(OneHotEncoder(handle_unknown='ignore',sparse_output=False),LinearRegression())),
            ]
            for k,descriptor in enumerate(DESCRIPTORS):
                baseline_specs.extend([
                    (f'individual_linear_{descriptor}',[k],make_pipeline(StandardScaler(),LinearRegression())),
                    (f'individual_forest_{descriptor}',[k],RandomForestRegressor(n_estimators=a.trees,max_depth=3,min_samples_leaf=3,random_state=seed,n_jobs=a.threads))])
            for name,columns,baseline in baseline_specs:
                baseline.fit(x(outer)[:,columns],y(outer))
                joblib.dump(dict(model=baseline,columns=columns),folder/f'{name}.joblib')
                record(name,baseline.predict(x(testing)[:,columns]))
            candidates = []
            for leaf in (1,3,5):
                for features in (1.,.67):
                    rf = RandomForestRegressor(n_estimators=a.trees,min_samples_leaf=leaf,max_features=features,random_state=seed,n_jobs=a.threads).fit(x(inner),y(inner))
                    candidates.append((mean_squared_error(y(validation),rf.predict(x(validation))),leaf,features))
            _,leaf,features = min(candidates)
            rf = RandomForestRegressor(n_estimators=a.trees,min_samples_leaf=leaf,max_features=features,random_state=seed,n_jobs=a.threads).fit(x(outer),y(outer))
            joblib.dump(rf,folder/'random_forest.joblib')
            record('random_forest',rf.predict(x(testing)))
            selections.append(dict(fold=fold,seed=seed,model='random_forest',selection=json.dumps(dict(min_samples_leaf=leaf,max_features=features))))
            neural_results = {}
            for kind,qs in [('ann',[0]+a.q),('gnn',a.q)]:
                for q in qs:
                    name = f'{kind}_q{q}' if q else 'ann'
                    model,scale,epochs,history = train_neural(inner,validation,kind,q,seed+1000*q+fold,a.epochs,a.patience)
                    val_prediction,_ = predict(model,validation,scale,kind)
                    val_r2 = r2_score(y(validation),val_prediction)
                    write_csv(folder/f'{name}_inner_curve.csv',history)
                    validation_scores = scores(y(validation), val_prediction)
                    fitted_cases = inner
                    if a.mode != 'v9':
                        model,scale,_,history = train_neural(outer,None,kind,q,seed+1000*q+fold,a.epochs,a.patience,fixed_epochs=epochs)
                        write_csv(folder/f'{name}_refit_curve.csv',history)
                        fitted_cases = outer
                    # v9 evaluates the restored best inner-fit checkpoint directly.
                    save_checkpoint(folder/f'{name}.pt',model,scale,kind,q,epochs,fitted_cases)
                    pred,latent = predict(model,testing,scale,kind)
                    # Latents are fold-specific: never pool axes across separately trained networks.
                    latent_rows = [dict(case=g['case'],**{f'q{k+1}':float(v) for k,v in enumerate(latent[j])}) for j,g in enumerate(testing)]
                    write_csv(folder/f'{name}_test_latent.csv',latent_rows)
                    if kind == 'gnn':
                        from interpret import latent_analysis
                        correlations,reconstruction=latent_analysis(model,scale,kind,fitted_cases,testing,seed,a.trees,a.threads)
                        write_csv(folder/f'{name}_correlations.csv',correlations)
                        write_csv(folder/f'{name}_descriptor_reconstruction.csv',reconstruction)
                    neural_results[(kind,q)] = (val_r2,pred)
                    selections.append(dict(fold=fold,seed=seed,model=name,selection=json.dumps(dict(best_epoch=epochs,inner_r2=float(val_r2),inner_rmse=validation_scores['rmse'],inner_mae=validation_scores['mae'],fit_cases=len(inner),validation_cases=len(validation),parameters=sum(p.numel() for p in model.parameters())))))
                    record(name,pred)
                if kind in ('gnn','ann'):
                    best = max(neural_results[(kind,q)][0] for q in a.q)
                    selected = min(q for q in a.q if neural_results[(kind,q)][0] >= best-.02)
                    record(f'{kind}_selected',neural_results[(kind,selected)][1])
                    selections.append(dict(fold=fold,seed=seed,model=f'{kind}_selected',selection=json.dumps(dict(q=selected,rule='smallest q within 0.02 inner validation R2 of best'))))
            write_csv(out/'selections.csv',selections)
            only_current=lambda entries:[r for r in entries if r['fold']==fold and r['seed']==seed]
            completed_state=dict(fold=fold,seed=seed,predictions=only_current(rows),metrics=only_current(metrics),selections=only_current(selections))
            temp=folder/'complete.tmp';temp.write_text(json.dumps(completed_state));temp.replace(folder/'complete.json')
    write_csv(out/'predictions.csv',rows)
    write_csv(out/'fold_metrics.csv',metrics)
    write_csv(out/'selections.csv',selections)
    provenance['status'] = 'complete'
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    from report import report
    report(out)
    from latent_selection_report import selection_report
    selection_report(out)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    p.add_argument('--q',nargs='+',type=int,default=[1,2,3,4],choices=[1,2,3,4])
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=50)
    p.add_argument('--trees',type=int,default=500)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--mode',choices=['grouped','method','size500','v9'],default='grouped')
    p.add_argument('--verify-splits',type=Path)
    p.add_argument('--particle-counts',nargs='+',type=int)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
