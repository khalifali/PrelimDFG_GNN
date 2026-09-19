"""Target-blind structural matching and exploratory paired descriptor analysis."""
import argparse, concurrent.futures, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'postprocessing'))
from exposure import geometric_exposure

BASIC=['box_dimension','hull_porosity','log_rg']
EXTRA=['asphericity','lambda2_over_lambda1','lambda3_over_lambda1','leaf_fraction','branch_fraction',
       'degree_std','mean_graph_distance','neighbors_1p5d','neighbors_2d','local_density_cv','core_shell_density_difference']

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def features(g):
    p=np.asarray(g['pos'],float);p-=p.mean(0);n=len(p)
    eig=np.linalg.eigvalsh(p.T@p/n)[::-1]
    edges=np.asarray(g['edge'],int);adj=np.zeros((n,n),bool);adj[edges[0],edges[1]]=True
    assert np.array_equal(adj,adj.T)
    degree=adj.sum(1);geodesic=shortest_path(csr_matrix(adj),directed=False,unweighted=True)
    assert np.isfinite(geodesic).all()
    dist=squareform(pdist(p));np.fill_diagonal(dist,np.inf)
    local=(dist<=2).sum(1);rad=np.linalg.norm(p,axis=1);rg=np.sqrt(np.mean(rad**2))
    inner=rad<=.5*rg;outer=rad>=rg
    core_shell=float(local[inner].mean()-local[outer].mean()) if inner.any() and outer.any() else 0.
    return dict(case=g['case'],n=n,group=g['independence_group'],target=float(g['target']),
        box_dimension=g['measured']['box_dimension'],hull_porosity=g['measured']['hull_porosity'],
        box_quality=g['measured']['box_quality'],box_grid_sd=g['measured']['box_grid_sd'],
        rg_over_d=float(rg),log_rg=float(np.log(rg)),contacts=int(adj.sum()/2),
        asphericity=float(1.5*np.sum(eig**2)/np.sum(eig)**2-.5),
        lambda2_over_lambda1=float(eig[1]/eig[0]),lambda3_over_lambda1=float(eig[2]/eig[0]),
        leaf_fraction=float(np.mean(degree==1)),branch_fraction=float(np.mean(degree>=3)),degree_std=float(np.std(degree)),
        mean_graph_distance=float(geodesic.sum()/(n*(n-1))),neighbors_1p5d=float((dist<=1.5).sum()/n),
        neighbors_2d=float(local.mean()),local_density_cv=float(local.std()/max(local.mean(),1e-12)),
        core_shell_density_difference=core_shell)

def match(rows,dmax,pmax,rg_limit=None,quality=False):
    """No target values are read here. One representative per known lineage."""
    representatives={}
    for r in sorted(rows,key=lambda x:x['case']):
        if quality and r['box_quality']!='ok':continue
        representatives.setdefault(r['group'],r)
    rr=list(representatives.values());graph=nx.Graph()
    for i,a in enumerate(rr):
        for b in rr[i+1:]:
            if a['n']!=b['n']:continue
            dd=abs(a['box_dimension']-b['box_dimension']);dp=abs(a['hull_porosity']-b['hull_porosity'])
            dr=abs(a['log_rg']-b['log_rg'])
            if dd>dmax or dp>pmax or (rg_limit and dr>np.log1p(rg_limit)):continue
            cost=(dd/dmax)**2+(dp/pmax)**2
            if rg_limit:cost+=(dr/np.log1p(rg_limit))**2
            graph.add_edge(a['case'],b['case'],weight=-cost,cost=cost)
    chosen=nx.max_weight_matching(graph,maxcardinality=True,weight='weight')
    result=[(min(a,b),max(a,b),graph[a][b]['cost']) for a,b in chosen]
    return sorted(result),graph.number_of_edges()

def pair_rows(rows,pairs):
    by={r['case']:r for r in rows};out=[]
    for a,b,cost in pairs:
        x,y=by[a],by[b]
        out.append(dict(a=a,b=b,n=x['n'],matching_distance=cost,delta_exposure=y['target']-x['target'],
            abs_delta_exposure=abs(y['target']-x['target']),both_box_quality_ok=x['box_quality']==y['box_quality']=='ok',
            **{'delta_'+k:y[k]-x[k] for k in BASIC+EXTRA}))
    return out

def summarize(label,pairs,eligible):
    gap=np.array([r['abs_delta_exposure'] for r in pairs])
    return dict(protocol=label,pairs=len(pairs),cases=2*len(pairs),eligible_edges=eligible,
        median_abs_exposure_gap=float(np.median(gap)) if len(gap) else None,
        p90_abs_exposure_gap=float(np.quantile(gap,.9)) if len(gap) else None,
        max_abs_exposure_gap=float(gap.max()) if len(gap) else None,
        median_abs_delta_Dbox=float(np.median([abs(r['delta_box_dimension']) for r in pairs])) if len(gap) else None,
        median_abs_delta_porosity=float(np.median([abs(r['delta_hull_porosity']) for r in pairs])) if len(gap) else None)

def paired_models(pairs,output):
    if len(pairs)<15:return []
    y=np.array([r['delta_exposure'] for r in pairs]);predictions=[];metrics=[]
    folds=list(KFold(n_splits=5,shuffle=True,random_state=9192026).split(y))
    for name,columns in [('zero',[]),('basic',BASIC),('expanded',BASIC+EXTRA)]:
        pred=np.zeros(len(y));alphas=[]
        if columns:
            x=np.array([[r['delta_'+c] for c in columns] for r in pairs])
            for fold,(train,test) in enumerate(folds):
                # Difference predictions are odd under exchanging pair members.
                pipe=make_pipeline(StandardScaler(with_mean=False),Ridge(fit_intercept=False))
                fit=GridSearchCV(pipe,{'ridge__alpha':[.01,.1,1,10,100,1000]},
                    cv=KFold(3,shuffle=True,random_state=1900+fold),scoring='neg_mean_squared_error')
                fit.fit(x[train],y[train]);pred[test]=fit.predict(x[test]);alphas.append(fit.best_params_['ridge__alpha'])
        rmse=float(np.sqrt(np.mean((pred-y)**2)));mae=float(np.mean(abs(pred-y)))
        metrics.append(dict(model=name,pairs=len(y),rmse=rmse,mae=mae,alphas=alphas))
        predictions.extend(dict(a=r['a'],b=r['b'],model=name,observed=float(yy),predicted=float(pp)) for r,yy,pp in zip(pairs,y,pred))
    write_csv(output/'paired_predictions.csv',predictions)
    return metrics

def analyze(source,output):
    output.mkdir(parents=True,exist_ok=False)
    graphs=json.loads(source.read_text())['records'];assert len(graphs)==359
    rows=[features(g) for g in graphs];write_csv(output/'structural_features.csv',rows)
    protocols=[('primary',.03,.01,None,False),('tight',.02,.005,None,False),('loose',.05,.02,None,False),
        ('rg_matched',.03,.01,.02,False),('quality_ok',.03,.01,None,True)]
    summaries=[];allpairs={}
    for label,d,p,rg,q in protocols:
        matching,eligible=match(rows,d,p,rg,q);pairs=pair_rows(rows,matching)
        # Regression guard: target perturbation cannot change any chosen pair.
        changed=[dict(r,target=-1234*i) for i,r in enumerate(rows)]
        assert matching==match(changed,d,p,rg,q)[0]
        used=[c for a,b,_ in matching for c in (a,b)];assert len(used)==len(set(used))
        write_csv(output/f'pairs_{label}.csv',pairs);allpairs[label]=pairs
        summaries.append(summarize(label,pairs,eligible))
    primary=allpairs['primary'];metrics=paired_models(primary,output)
    associations=[]
    if len(primary)>3:
        y=[r['abs_delta_exposure'] for r in primary]
        for col in BASIC+EXTRA:
            x=[abs(r['delta_'+col]) for r in primary]
            rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 else None
            associations.append(dict(feature=col,spearman_abs_pair_difference=rho))
    write_csv(output/'exploratory_associations.csv',associations)
    selected=sorted(primary,key=lambda x:-x['abs_delta_exposure'])[:3]
    chosen={(r['a'],r['b']) for r in selected}
    for r in sorted(primary,key=lambda x:x['matching_distance']):
        if len(selected)>=6:break
        if (r['a'],r['b']) not in chosen:selected.append(r);chosen.add((r['a'],r['b']))
    (output/'diagnostic_pairs.json').write_text(json.dumps(selected,indent=2))
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=len(rows),
        source_graph_coordinates='archived centered/diameter-normalized float32 coordinates; ray reproduction checked separately',
        status='exploratory development analysis',summaries=summaries,paired_models=metrics,
        diagnostic_selection='Three largest exposure gaps, then three closest descriptor matches not already chosen',
        numerical_checks='pending')
    (output/'summary.json').write_text(json.dumps(result,indent=2));report(output)
    print(json.dumps(result,indent=2),flush=True)

def ray_case(g):
    p=np.asarray(g['pos'],float);r=np.full(len(p),.5)
    values={str(n):float(geometric_exposure(p,r,n).mean()) for n in [2048,8192,16384]}
    q,_=np.linalg.qr(np.random.default_rng(91933).normal(size=(3,3)))
    values['8192_rotated']=float(geometric_exposure(p@q,r,8192).mean())
    vals=list(values.values())
    out=dict(case=g['case'],archived=float(g['target']),**values,
        reproduction_error=abs(values['2048']-g['target']),observed_numerical_spread=max(vals)-min(vals))
    print(json.dumps(out),flush=True);return out

def numerics(source,output):
    pairs=json.loads((output/'diagnostic_pairs.json').read_text());names={r[k] for r in pairs for k in ['a','b']}
    graphs=json.loads(source.read_text())['records'];chosen=[g for g in graphs if g['case'] in names]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(ray_case,g) for g in chosen]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result());write_csv(output/'ray_checks.csv',rows)
    by={r['case']:r for r in rows};checks=[]
    for pair in pairs:
        a,b=by[pair['a']],by[pair['b']]
        gaps=[b[k]-a[k] for k in ['2048','8192','16384','8192_rotated']]
        spread=a['observed_numerical_spread']+b['observed_numerical_spread']
        checks.append(dict(a=pair['a'],b=pair['b'],archived_abs_gap=pair['abs_delta_exposure'],
            gap_16384=abs(gaps[2]),sum_observed_numerical_spreads=spread,
            gap_exceeds_observed_spread=abs(gaps[2])>spread,
            sign_consistent=all(x>0 for x in gaps) or all(x<0 for x in gaps),
            reproduction_ok=max(a['reproduction_error'],b['reproduction_error'])<=1e-4))
    write_csv(output/'pair_numerical_checks.csv',checks)
    result=json.loads((output/'summary.json').read_text());result['numerical_checks']=checks
    (output/'summary.json').write_text(json.dumps(result,indent=2));report(output)

def report(output):
    result=json.loads((output/'summary.json').read_text())
    lines=['# Matched-structure development analysis','',
      'Existing 359 BPM geometries. No new GNN training. FracVAL generation remains paused.',
      'Matching is target-blind, maximum-cardinality then minimum descriptor distance, and disjoint within each protocol.',
      'Primary tolerances: identical N, |delta Dbox| <= 0.03, |delta hull porosity| <= 0.01.',
      'Pairs match estimated descriptors, not proven identical physical morphology. Quality flags and residual mismatch matter.','',
      '| Matching | Pairs | Median exposure gap | 90th percentile gap | Maximum gap |',
      '|---|---:|---:|---:|---:|']
    for s in result['summaries']:
        lines.append('| '+ ' | '.join(str(s[k]) for k in ['protocol','pairs','median_abs_exposure_gap','p90_abs_exposure_gap','max_abs_exposure_gap'])+' |')
    lines+=['','## Can additional descriptors predict paired exposure differences?','',
      'Nested pair-level cross-validation of signed differences; no pair member appears in another fold. Basic includes residual Dbox, porosity and log Rg differences. Expanded adds anisotropy, branching and local-density descriptors.',
      'This is a small exploratory linear comparison, not a definitive test of nonlinear sufficiency or GNN advantage.','',
      '| Model | Held-out difference RMSE | MAE |','|---|---:|---:|']
    for r in result['paired_models']:lines.append(f"| {r['model']} | {r['rmse']:.6f} | {r['mae']:.6f} |")
    lines+=['','## Numerical check','',
      'Diagnostic pairs: three largest observed gaps and three closest descriptor matches. This selection is disclosed and is not a random error sample.',
      'Ray checks use archived float32 graph coordinates. Agreement with archived 2048-ray labels is checked before interpretation.',
      'Observed variation across ray counts/orientation is a sensitivity diagnostic, not a rigorous error bound.']
    if isinstance(result['numerical_checks'],list):
        lines+=['','| Pair | 16384-ray absolute gap | Sum of observed numerical spreads | Reproduces archive |','|---|---:|---:|---|']
        for r in result['numerical_checks']:lines.append(f"| {r['a']} / {r['b']} | {r['gap_16384']:.6f} | {r['sum_observed_numerical_spreads']:.6f} | {r['reproduction_ok']} |")
    else:lines+=['','Ray checks pending.']
    lines+=['','Raw signed/absolute pair differences, all features, model predictions and exploratory associations are retained as CSV. Associations alone are not causal explanations.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['analyze','numerics']);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (analyze if a.phase=='analyze' else numerics)(a.source,a.output)
