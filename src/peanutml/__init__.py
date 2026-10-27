"""Reusable transcriptomic ML workflow derived from Frederick Mensah's notebook."""
__version__ = '0.1.1'
from .data import Dataset, load_dataset
from .models import MODEL_KEYS, get_model
from .evaluation import make_cv, run_nested_cv
from .selectors import PCAGeneSelector
from .consensus import consensus_genes, hyperparameter_frequency
from .final_model import FROZEN_GENES, ModelBundle, fit_final, validate_external, bootstrap_ci
