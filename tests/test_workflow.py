import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
from peanutml import *

@pytest.fixture
def toy():
    rng=np.random.default_rng(19); y=np.tile([0,1],20)
    X=pd.DataFrame(rng.normal(size=(40,12)),columns=[f'g{i}' for i in range(12)])
    X.iloc[:,:3]+=y[:,None]
    return X,y,pd.Series([f's{i}' for i in range(40)])

@pytest.mark.parametrize('name',MODEL_KEYS)
def test_all_pipelines_nested_cv(name,toy,tmp_path):
    X,y,ids=toy; pipe,grid=get_model(name,[3])
    grid={k:[v[0]] for k,v in grid.items()}
    if 'clf__n_estimators' in grid:grid['clf__n_estimators']=[10]
    outer,inner=make_cv(2,1,2,1)
    result=run_nested_cv(name,pipe,grid,X,y,ids,outer,inner,output_dir=tmp_path)
    assert len(result['fold_results'])==2
    assert len(result['predictions'])==40
    assert not result['predictions'].Sample_ID.duplicated().any()
    assert result['gene_frequency'].Selection_Percentage.max()<=100
    assert len(list(tmp_path.glob('*.csv')))==6
    assert not hasattr(pipe.named_steps['clf'],'classes_') # template not fitted

def test_original_cv_equivalence(toy,tmp_path,monkeypatch):
    # Run the actual notebook function on 50 small outer folds to preserve its denominator.
    X,y,ids=toy
    source=json.loads((Path(__file__).parents[1]/'provenance/source_notebook.ipynb').read_text())
    text=''.join(source['cells'][7]['source']);node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef))
    scope=dict(np=np,pd=pd,json=json,balanced_accuracy_score=balanced_accuracy_score,
       f1_score=f1_score,matthews_corrcoef=matthews_corrcoef,roc_auc_score=roc_auc_score)
    def serial_grid(*args,**kwargs):kwargs['n_jobs']=1;return GridSearchCV(*args,**kwargs)
    scope['GridSearchCV']=serial_grid
    exec(ast.get_source_segment(text,node),scope)
    pipe,grid=get_model('anova_lr',[3]);grid['clf__C']=[1];grid['clf__class_weight']=[None]
    outer,inner=make_cv(5,10,2,1)
    monkeypatch.chdir(tmp_path)
    original=scope['run_nested_cv']('LR',pipe,grid,X,y,ids,outer,inner)
    packaged=run_nested_cv('LR',pipe,grid,X,y,ids,outer,inner)
    for key in original:pd.testing.assert_frame_equal(original[key],packaged[key])

def test_pca_matches_notebook(toy):
    X,y,_=toy
    source=json.loads((Path(__file__).parents[1]/'provenance/source_notebook.ipynb').read_text())
    scope={'np':np};exec(''.join(source['cells'][13]['source']),scope)
    original=scope['PCAGeneSelector'](k=3).fit(X.iloc[:25])
    packaged=PCAGeneSelector(k=3).fit(X.iloc[:25])
    np.testing.assert_array_equal(original.get_support(),packaged.get_support())
    np.testing.assert_allclose(original.transform(X),packaged.transform(X))

def test_frozen_alignment_and_no_refit(toy,tmp_path):
    import joblib
    X,y,ids=toy;bundle=fit_final(X,y,genes=['g2','g0','g1'])
    before=bundle.estimator.named_steps['scale'].mean_.copy()
    summary,preds=validate_external(bundle,X,y,ids)
    a=bundle.predict(X)[1];b=bundle.predict(X[X.columns[::-1]])[1]
    np.testing.assert_allclose(a,b)
    np.testing.assert_array_equal(before,bundle.estimator.named_steps['scale'].mean_)
    joblib.dump(bundle,tmp_path/'model.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'model.joblib').predict(X)[1],a)
    with pytest.raises(ValueError,match='Missing genes'):bundle.predict(X.drop(columns=['g0']))

def test_data_validation(toy,tmp_path):
    X,y,ids=toy;df=X.assign(Condition=np.where(y,'Drought','Control'),Batch='b')
    df.insert(0,'Unnamed: 0',ids)
    path=tmp_path/'data.csv';df.to_csv(path,index=False)
    d=load_dataset(path);pd.testing.assert_frame_equal(d.X,X)
    df.loc[0,'Condition']='typo';df.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Unknown'):load_dataset(path)
    df.loc[0,'Condition']='Control';df.loc[0,'g0']=np.nan;df.to_csv(path,index=False)
    with pytest.raises(ValueError,match='missing'):load_dataset(path)
    path.write_text('SampleID,Condition,g,g\na,Control,1,2\nb,Drought,2,3\n')
    with pytest.raises(ValueError,match='duplicate'):load_dataset(path,id_col='SampleID')

def test_consensus():
    a=pd.DataFrame({'Gene':['a','b'],'Selection_Percentage':[100,50]})
    b=pd.DataFrame({'Gene':['a','c'],'Selection_Percentage':[50,100]})
    _,panel=consensus_genes({'one':a,'two':b})
    assert panel.Gene.tolist()==['a']

def test_bootstrap_and_tuning(toy):
    X,y,ids=toy;_,inner=make_cv(2,1,2,1)
    model=fit_final(X,y,genes=['g0','g1'],mode='tuned',inner_cv=inner)
    pred,score=model.predict(X)
    first=bootstrap_ci(y,pred,score,n_boot=50)
    pd.testing.assert_frame_equal(first,bootstrap_ci(y,pred,score,n_boot=50))
    assert len(first)==4
