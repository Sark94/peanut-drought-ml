"""Run the three original consensus models. Full grids can take substantial time."""
import argparse
from pathlib import Path
from peanutml import load_dataset,get_model,make_cv,run_nested_cv,consensus_genes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv');p.add_argument('--output',default='consensus_results');p.add_argument('--jobs',type=int,default=1)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=load_dataset(args.csv);outer,inner=make_cv();freq={}
    for name in ['anova_lr','anova_linear_svm','pca_linear_svm']:
        pipe,grid=get_model(name)
        result=run_nested_cv(name,pipe,grid,data.X,data.y,data.sample_ids,outer,inner,output_dir=out/name,n_jobs=args.jobs)
        freq[name]=result['gene_frequency']
    comparison,strict=consensus_genes(freq)
    _,robust=consensus_genes(freq,minimum_pipelines=2)
    comparison.to_csv(out/'comparison.csv',index=False)
    strict.to_csv(out/'strict_panel.csv',index=False)
    robust.to_csv(out/'robust_panel.csv',index=False)
    rows=[]
    for threshold in [40,50,60,70]:
        for minimum in [1,2,3]:
            _,panel=consensus_genes(freq,threshold,minimum)
            rows.append({'Threshold':threshold,'Minimum_Pipelines':minimum,'N_Genes':len(panel)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'threshold_sensitivity.csv',index=False)

if __name__=='__main__':main()
