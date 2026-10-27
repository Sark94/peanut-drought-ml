"""Original nine model definitions and full parameter grids."""
from .selectors import PCAGeneSelector
RANDOM_STATE = 42
# ==========================================
# Cell 6: Define ANOVA pipelines and grids
# ==========================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

K_VALUES = [5, 10, 15, 20, 30, 40, 50]

# Logistic Regression
pipe_lr = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

param_lr = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"]
}

# Linear SVM
pipe_lsvm = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
])

param_lsvm = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"]
}

# RBF-SVM
pipe_rbf = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
])

param_rbf = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "clf__class_weight": [None, "balanced"]
}

# Random Forest
pipe_rf = Pipeline([
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
])

param_rf = {
    "select__k": K_VALUES,
    "clf__n_estimators": [200, 500],
    "clf__max_depth": [None, 3, 5, 10],
    "clf__min_samples_split": [2, 4, 6],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
    "clf__class_weight": [None, "balanced"]
}

# KNN
pipe_knn = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", KNeighborsClassifier())
])

param_knn = {
    "select__k": K_VALUES,
    "clf__n_neighbors": [1, 3, 5, 7, 9],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"]
}

# ==========================================
# Cell 14: Define PCA-based pipelines
# ==========================================

# Logistic Regression
pipe_pca_lr = Pipeline([
    ("select", PCAGeneSelector(variance_threshold=0.80)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

param_pca_lr = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"]
}

# Linear SVM
pipe_pca_lsvm = Pipeline([
    ("select", PCAGeneSelector(variance_threshold=0.80)),
    ("scale", StandardScaler()),
    ("clf", SVC(
        kernel="linear",
        probability=True,
        random_state=RANDOM_STATE
    ))
])

param_pca_lsvm = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"]
}

# RBF-SVM
pipe_pca_rbf = Pipeline([
    ("select", PCAGeneSelector(variance_threshold=0.80)),
    ("scale", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ))
])

param_pca_rbf = {
    "select__k": K_VALUES,
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "clf__class_weight": [None, "balanced"]
}

# KNN
pipe_pca_knn = Pipeline([
    ("select", PCAGeneSelector(variance_threshold=0.80)),
    ("scale", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

param_pca_knn = {
    "select__k": K_VALUES,
    "clf__n_neighbors": [1, 3, 5, 7, 9],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"]
}


MODEL_KEYS = ('anova_lr', 'anova_linear_svm', 'anova_rbf_svm', 'anova_rf', 'anova_knn',
              'pca_lr', 'pca_linear_svm', 'pca_rbf_svm', 'pca_knn')

def get_model(name, k_values=None, random_state=42):
    """Return a fresh unfitted pipeline and independent grid; defaults match notebook."""
    from sklearn.base import clone
    from copy import deepcopy
    pairs = [(pipe_lr,param_lr), (pipe_lsvm,param_lsvm), (pipe_rbf,param_rbf),
             (pipe_rf,param_rf), (pipe_knn,param_knn), (pipe_pca_lr,param_pca_lr),
             (pipe_pca_lsvm,param_pca_lsvm), (pipe_pca_rbf,param_pca_rbf),
             (pipe_pca_knn,param_pca_knn)]
    if name not in MODEL_KEYS:
        raise ValueError(f'Unknown model {name!r}; choose from {MODEL_KEYS}')
    pipe, grid = pairs[MODEL_KEYS.index(name)]
    pipe, grid = clone(pipe), deepcopy(grid)
    if 'clf__random_state' in pipe.get_params():
        pipe.set_params(clf__random_state=random_state)
    if k_values is not None:
        values = list(k_values)
        if not values or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in values):
            raise ValueError('k_values must contain positive integers')
        grid['select__k'] = values
    return pipe, grid
