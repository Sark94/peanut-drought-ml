"""Cross-pipeline feature recurrence and parameter frequency summaries."""
from collections import Counter
import json
import pandas as pd

def consensus_genes(frequencies, threshold=50, minimum_pipelines=None):
    """Merge {pipeline_name: gene_frequency_table}; return comparison and selected panel."""
    if not frequencies or not 0<=threshold<=100:
        raise ValueError('Provide frequency tables and a threshold between 0 and 100')
    minimum_pipelines=len(frequencies) if minimum_pipelines is None else minimum_pipelines
    if not 1<=minimum_pipelines<=len(frequencies):
        raise ValueError('minimum_pipelines outside available pipelines')
    comparison=None
    columns=[]
    for name,table in frequencies.items():
        column=f'{name}_Percent'; columns.append(column)
        if table.Gene.duplicated().any():raise ValueError('Duplicate genes in frequency table')
        subset=table[['Gene','Selection_Percentage']].rename(columns={'Selection_Percentage':column})
        comparison=subset if comparison is None else comparison.merge(subset,on='Gene',how='outer')
    comparison=comparison.fillna(0)
    comparison['Pipelines_Selected']=(comparison[columns]>0).sum(axis=1)
    comparison['Pipelines_Above_Threshold']=(comparison[columns]>=threshold).sum(axis=1)
    comparison['Mean_Selection_Percent']=comparison[columns].mean(axis=1)
    comparison=comparison.sort_values(['Mean_Selection_Percent','Gene'],ascending=[False,True]).reset_index(drop=True)
    selected=comparison[comparison.Pipelines_Above_Threshold>=minimum_pipelines].copy()
    return comparison,selected

def hyperparameter_frequency(best_params):
    """Count combinations without accidentally grouping by outer fold number."""
    counts=Counter(json.dumps(json.loads(s) if isinstance(s,str) else s,sort_keys=True) for s in best_params.Best_Params)
    return pd.DataFrame([{**json.loads(k),'Selection_Count':v,'Selection_Percent':100*v/len(best_params)} for k,v in counts.most_common()])
