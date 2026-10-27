from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from peanutml import *
from peanutml.plotting import plot_confusion
root=Path(__file__).resolve().parents[1]
inputs=root/'provenance/data'
out=root/'docs/real_data_checks';out.mkdir(parents=True,exist_ok=True)
d=load_dataset(inputs/'sark_276_ComBatSeq_VST_for_ML_with_condition.csv')
fixed=fit_final(d.X,d.y,mode='fixed')
tuned=fit_final(d.X,d.y,mode='tuned')
assert tuned.specification['best_params']=={'clf__C':10,'clf__class_weight':None}
original=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(penalty='l2',C=10,class_weight=None,solver='liblinear',max_iter=5000,random_state=42))])
original.fit(d.X[list(FROZEN_GENES)],d.y)
rows=[]
for name,counts in [('Jiang',[6,0,1,17]),('Sun',[6,0,2,10])]:
    e=load_dataset(inputs/f'{name}_external_frozen8_VST.csv',id_col='SampleID')
    summary,pred=validate_external(fixed,e.X,e.y,e.sample_ids,dataset_name=name)
    np.testing.assert_allclose(fixed.predict(e.X)[1],original.predict_proba(e.X[list(FROZEN_GENES)])[:,1],rtol=1e-12)
    np.testing.assert_allclose(fixed.predict(e.X)[1],tuned.predict(e.X)[1],rtol=1e-12)
    assert summary[['TN','FP','FN','TP']].iloc[0].tolist()==counts
    summary.to_csv(out/f'{name}_summary.csv',index=False)
    pred.to_csv(out/f'{name}_predictions.csv',index=False)
    plot_confusion(e.y,pred.Predicted_Label,out/f'{name}_confusion.png',title=name)
    rows.append(summary)
pd.concat(rows).to_csv(out/'external_summary.csv',index=False)
report={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
        'discovery_shape':list(d.X.shape),'tuned_parameters':tuned.specification['best_params'],
        'final_model_predictions_match_original':True,'fixed_and_tuned_predictions_match':True,
        'external_counts_match_notebook_outputs':True,
        'full_nine_model_nested_cv_rerun':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()}}
(out/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
