"""Command-line entry point: nested CV, final training, external evaluation."""
import argparse
import json
from pathlib import Path
import joblib
from . import load_dataset,get_model,make_cv,run_nested_cv,fit_final,validate_external,bootstrap_ci,MODEL_KEYS

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    for command in ('cv','train','validate'):
        a=subs.add_parser(command)
        a.add_argument('csv'); a.add_argument('--output',required=True)
        a.add_argument('--id-column',default='Unnamed: 0' if command!='validate' else 'SampleID')
        a.add_argument('--label-column',default='Condition')
        a.add_argument('--batch-column',default='Batch')
        if command!='validate':
            a.add_argument('--positive-label',default='Drought'); a.add_argument('--negative-label',default='Control')
            a.add_argument('--jobs',type=int,default=1)
            a.add_argument('--seed',type=int,default=42)
        if command=='cv':
            a.add_argument('--model',choices=MODEL_KEYS,default='anova_lr')
            a.add_argument('--outer-splits',type=int,default=5);a.add_argument('--outer-repeats',type=int,default=10)
            a.add_argument('--inner-splits',type=int,default=4);a.add_argument('--inner-repeats',type=int,default=3)
            a.add_argument('--k',type=int,nargs='+')
        elif command=='train':
            a.add_argument('--mode',choices=['fixed','tuned'],required=True)
            a.add_argument('--genes',help='CSV containing ordered Gene column; defaults to notebook eight-gene panel')
        else:
            a.add_argument('--model-file',required=True,help='Trusted local joblib bundle only')
            a.add_argument('--bootstrap',type=int,default=5000)
    args=p.parse_args(argv)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.command=='validate':
        # joblib is pickle-based: use only trusted artifacts.
        bundle=joblib.load(args.model_file)
        d=load_dataset(args.csv,id_col=args.id_column,label_col=args.label_column,batch_col=args.batch_column,
            negative_label=bundle.classes[0],positive_label=bundle.classes[1],genes=bundle.genes)
        summary,preds=validate_external(bundle,d.X,d.y,d.sample_ids)
        summary.to_csv(out/'summary.csv',index=False);preds.to_csv(out/'predictions.csv',index=False)
        bootstrap_ci(d.y,preds.Predicted_Label,preds.Positive_Probability,n_boot=args.bootstrap).to_csv(out/'confidence_intervals.csv',index=False)
        from .plotting import plot_confusion
        plot_confusion(d.y,preds.Predicted_Label,out/'confusion_matrix.png',classes=bundle.classes)
    else:
        d=load_dataset(args.csv,id_col=args.id_column,label_col=args.label_column,batch_col=args.batch_column,
            positive_label=args.positive_label,negative_label=args.negative_label)
        if args.command=='cv':
            pipe,grid=get_model(args.model,args.k,args.seed)
            if max(grid['select__k'])>d.X.shape[1]:p.error('k exceeds gene count; supply --k')
            outer,inner=make_cv(args.outer_splits,args.outer_repeats,args.inner_splits,args.inner_repeats,args.seed)
            run_nested_cv(args.model,pipe,grid,d.X,d.y,d.sample_ids,outer,inner,output_dir=out,n_jobs=args.jobs)
        else:
            kwargs={}
            if args.genes:
                import pandas as pd
                kwargs['genes']=pd.read_csv(args.genes).Gene.tolist()
            bundle=fit_final(d.X,d.y,mode=args.mode,n_jobs=args.jobs,random_state=args.seed,classes=d.classes,**kwargs)
            joblib.dump(bundle,out/'model.joblib')
            (out/'model_specification.json').write_text(json.dumps(bundle.specification,indent=2))
    import platform,sklearn,numpy,pandas
    (out/'run_configuration.json').write_text(json.dumps({**vars(args),'python':platform.python_version(),
        'sklearn':sklearn.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__},indent=2))
    print(f'Results saved to {out}')

if __name__=='__main__':main()
