"""Fold-specific latent correlations and descriptor-to-latent reconstruction."""
import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset import DESCRIPTORS
from models import predict


def latent_analysis(model,scale,kind,training,testing,seed,trees,threads):
    _,fit_latent=predict(model,training,scale,kind)
    _,test_latent=predict(model,testing,scale,kind)
    fit_x=np.stack([g['descriptors'] for g in training])
    test_x=np.stack([g['descriptors'] for g in testing])
    correlations=[]
    for partition,data,latent in [('train',training,fit_latent),('test',testing,test_latent)]:
        values=np.column_stack([np.stack([g['descriptors'] for g in data]),[g['target'] for g in data]])
        for j in range(latent.shape[1]):
            for k,name in enumerate(DESCRIPTORS+['geometric_exposure']):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pearson=pearsonr(values[:,k],latent[:,j]).statistic
                    spearman=spearmanr(values[:,k],latent[:,j]).statistic
                correlations.append(dict(partition=partition,coordinate=j+1,descriptor=name,pearson=float(pearson),spearman=float(spearman)))
    reconstruction=[]
    for name,columns,estimator in [('six_linear',list(range(6)),make_pipeline(StandardScaler(),LinearRegression())),
            ('six_random_forest',list(range(6)),RandomForestRegressor(n_estimators=trees,max_depth=4,min_samples_leaf=3,random_state=seed,n_jobs=threads))]+[
            (f'linear_{name}',[k],make_pipeline(StandardScaler(),LinearRegression())) for k,name in enumerate(DESCRIPTORS)]:
        estimator.fit(fit_x[:,columns],fit_latent)
        prediction=estimator.predict(test_x[:,columns]).reshape(test_latent.shape)
        for j in range(test_latent.shape[1]):
            reconstruction.append(dict(model=name,coordinate=j+1,r2=float(r2_score(test_latent[:,j],prediction[:,j])),rmse=float(np.sqrt(mean_squared_error(test_latent[:,j],prediction[:,j])))))
    return correlations,reconstruction
