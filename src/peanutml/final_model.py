"""Fit and apply a frozen feature panel; external evaluation never refits."""
from dataclasses import dataclass
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score,
                            matthews_corrcoef, roc_auc_score, confusion_matrix)
from .data import validate_xy

FROZEN_GENES = ('LOC112784224','LOC112733679','LOC112775500','LOC112795505',
                'LOC112720488','LOC112741057','LOC112755969','LOC112790749')

@dataclass
class ModelBundle:
    estimator: object
    genes: tuple
    classes: tuple
    specification: dict

    def predict(self, X):
        """Align genes by name, use discovery scaler, return encoded label and positive score."""
        if not isinstance(X,pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError('Use a DataFrame with unique gene columns')
        missing=set(self.genes)-set(X.columns)
        if missing:raise ValueError(f'Missing genes: {sorted(missing)}')
        values=X.loc[:,list(self.genes)]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite values in prediction input')
        labels=self.estimator.predict(values)
        scores=self.estimator.predict_proba(values)[:,1]
        return labels,scores

def fit_final(X,y,genes=FROZEN_GENES,*,mode='fixed',inner_cv=None,n_jobs=1,
              random_state=42,classes=('Control','Drought')):
    """Fit fixed C=10 LR (cell 26), or tune LR on supplied panel (cell 25).

    `tuned` reproduces cell 25 only when the same ordered consensus genes are supplied.
    best CV score after full-discovery feature selection is not an unbiased test score.
    """
    validate_xy(X,y)
    genes=tuple(genes)
    if not genes or len(genes)!=len(set(genes)):raise ValueError('Panel must contain unique genes')
    if len(classes)!=2 or classes[0]==classes[1]:raise ValueError('Provide two distinct class names')
    missing=set(genes)-set(X.columns)
    if missing:raise ValueError(f'Missing panel genes: {sorted(missing)}')
    pipe=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(
        penalty='l2',solver='liblinear',max_iter=5000,random_state=random_state,C=10))])
    spec={'mode':mode,'genes':list(genes),'classes':list(classes),'random_state':random_state,
          'discovery_samples':len(X),'python':platform.python_version(),'sklearn':sklearn.__version__}
    if mode=='fixed':
        fitted=pipe.fit(X.loc[:,list(genes)],y)
    elif mode=='tuned':
        if inner_cv is None:
            from .evaluation import make_cv
            _,inner_cv=make_cv(random_state=random_state)
        search=GridSearchCV(pipe,{'clf__C':[0.01,0.1,1,10,100],
            'clf__class_weight':[None,'balanced']},scoring='balanced_accuracy',
            cv=inner_cv,n_jobs=n_jobs,refit=True,error_score='raise')
        search.fit(X.loc[:,list(genes)],y)
        fitted=search.best_estimator_
        spec.update(best_params=search.best_params_,discovery_tuning_score=search.best_score_)
    else:raise ValueError('mode must be fixed or tuned')
    spec['classifier_parameters']=fitted.named_steps['clf'].get_params()
    return ModelBundle(fitted,genes,tuple(classes),spec)

def metric_values(y,pred,score):
    """Binary evaluation; callers must supply both outcome classes."""
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'Balanced_Accuracy':balanced_accuracy_score(y,pred),'Accuracy':accuracy_score(y,pred),
            'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred),
            'ROC_AUC':roc_auc_score(y,score),'Sensitivity':tp/(tp+fn),
            'Specificity':tn/(tn+fp),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def validate_external(bundle,X,y,sample_ids=None,dataset_name='external'):
    """Evaluate frozen model on already-preprocessed, independently held-out inputs."""
    validate_xy(X,y,sample_ids)
    pred,score=bundle.predict(X)
    summary=pd.DataFrame([{'Dataset':dataset_name,'N':len(y),**metric_values(y,pred,score)}])
    records=pd.DataFrame({'SampleID':np.arange(len(y)) if sample_ids is None else np.asarray(sample_ids),
        'True_Label':y,'Predicted_Label':pred,
        'True_Condition':[bundle.classes[int(v)] for v in y],
        'Predicted_Condition':[bundle.classes[int(v)] for v in pred],
        'Positive_Probability':score})
    return summary,records

def bootstrap_ci(y,pred,score,n_boot=5000,random_state=42,rng=None):
    """Notebook's unstratified percentile bootstrap; single-class resamples skipped.

    Pass a shared np.random.Generator to reproduce sequential cohort RNG use.
    These intervals condition on the fitted model, not training/model-selection uncertainty.
    """
    y,pred,score=map(np.asarray,(y,pred,score))
    if len(y)!=len(pred) or len(y)!=len(score) or set(y.tolist())!={0,1}:
        raise ValueError('Aligned binary labels and both true classes required')
    if not np.isfinite(score).all() or not isinstance(n_boot,int) or n_boot<1:
        raise ValueError('Finite scores and a positive integer n_boot required')
    rng=np.random.default_rng(random_state) if rng is None else rng
    functions={'Balanced_Accuracy':lambda a,b,c:balanced_accuracy_score(a,b),
               'F1':lambda a,b,c:f1_score(a,b,zero_division=0),
               'MCC':lambda a,b,c:matthews_corrcoef(a,b),
               'ROC_AUC':lambda a,b,c:roc_auc_score(a,c)}
    rows=[]
    for name,func in functions.items():
        vals=[]
        for _ in range(n_boot):
            idx=rng.integers(0,len(y),len(y))
            if len(np.unique(y[idx]))<2:continue
            vals.append(func(y[idx],pred[idx],score[idx]))
        if not vals:raise ValueError('No valid bootstrap replicates; increase n_boot')
        low,high=np.percentile(vals,[2.5,97.5])
        rows.append({'Metric':name,'Estimate':func(y,pred,score),'CI_Lower_95':low,
                     'CI_Upper_95':high,'Bootstrap_Replicates_Used':len(vals)})
    return pd.DataFrame(rows)
