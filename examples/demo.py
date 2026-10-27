"""Small synthetic example, not evidence of biological performance."""
import numpy as np
import pandas as pd
from peanutml import get_model, make_cv, run_nested_cv
rng=np.random.default_rng(42)
y=np.tile([0,1],20)
X=pd.DataFrame(rng.normal(size=(40,12)),columns=[f'gene_{i}' for i in range(12)])
X.iloc[:,:3]+=y[:,None]
pipe,grid=get_model('anova_lr',[3,5])
grid['clf__C']=[0.1,1];grid['clf__class_weight']=[None]
outer,inner=make_cv(outer_splits=3,outer_repeats=1,inner_splits=2,inner_repeats=1)
run_nested_cv('demo_lr',pipe,grid,X,y,pd.Series([f'sample_{i}' for i in range(40)]),
              outer,inner,output_dir='demo_results')
