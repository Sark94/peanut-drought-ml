"""Validated sample-by-gene inputs; raw-count normalization is outside this package."""
from dataclasses import dataclass
import csv
import numpy as np
import pandas as pd

@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    sample_ids: pd.Series
    classes: tuple
    batch: pd.Series | None = None

def validate_xy(X, y, sample_ids=None):
    if not isinstance(X,pd.DataFrame) or X.empty:
        raise ValueError('X must be a nonempty sample-by-gene DataFrame')
    if X.columns.duplicated().any():
        raise ValueError('Duplicate gene names')
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError('Predictors must be numeric; exclude sample metadata')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Predictors contain missing or infinite values')
    if len(y)!=len(X) or set(np.asarray(y).tolist())!={0,1}:
        raise ValueError('Both binary labels 0 and 1 are required and must align with X')
    if sample_ids is not None and (len(sample_ids)!=len(X) or pd.Series(sample_ids).isna().any() or pd.Series(sample_ids).duplicated().any()):
        raise ValueError('Sample IDs must be nonmissing, unique, and aligned with X')

def load_dataset(path, label_col='Condition', id_col='Unnamed: 0', batch_col='Batch',
                 positive_label='Drought', negative_label='Control', genes=None):
    """Read a CSV, exclude ID/label/batch, map explicit class names, reject invalid data."""
    with open(path, newline='',encoding='utf-8-sig') as handle:
        header=next(csv.reader(handle))
    if len(header)!=len(set(header)):
        raise ValueError('CSV has duplicate column names')
    df=pd.read_csv(path)
    if label_col not in df or id_col not in df:
        raise ValueError(f'Required columns: {label_col!r}, {id_col!r}')
    if positive_label==negative_label:
        raise ValueError('Class names must differ')
    mapped=df[label_col].map({negative_label:0, positive_label:1})
    if mapped.isna().any():
        raise ValueError('Unknown or missing outcome labels')
    excluded=[id_col,label_col]+([batch_col] if batch_col in df else [])
    X=df.drop(columns=excluded) if genes is None else df.loc[:,list(genes)]
    y=mapped.to_numpy(dtype=int)
    validate_xy(X,y,df[id_col])
    return Dataset(X,y,df[id_col].copy(),(negative_label,positive_label),
                   df[batch_col].copy() if batch_col in df else None)
