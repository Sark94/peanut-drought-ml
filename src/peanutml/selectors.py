"""PCA loading-based gene selection from the source notebook."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAGeneSelector(BaseEstimator, TransformerMixin):
    """
    Rank genes using PCA loadings calculated from training data only.

    Steps:
    1. Standardize training data.
    2. Fit PCA retaining enough components to explain >=80% variance.
    3. Calculate gene importance:
       sum(|loading| × explained variance ratio)
       across retained PCs.
    4. Retain the top k genes.
    """

    def __init__(self, k=20, variance_threshold=0.80):
        self.k = k
        self.variance_threshold = variance_threshold

    def fit(self, X, y=None):

        # Preserve feature names when X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f"feature_{i}" for i in range(X.shape[1])]
            )

        # Standardize TRAINING data only
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Retain minimum PCs explaining ~80% cumulative variance
        self.pca_ = PCA(
            n_components=self.variance_threshold,
            svd_solver="full"
        )

        self.pca_.fit(X_scaled)

        # PCA loadings: genes × retained PCs
        loadings = self.pca_.components_.T

        # Weight absolute loadings by variance explained by each PC
        self.gene_scores_ = np.sum(
            np.abs(loadings) * self.pca_.explained_variance_ratio_,
            axis=1
        )

        # Rank genes from highest to lowest PCA importance
        ranked_indices = np.argsort(self.gene_scores_)[::-1]

        # Retain top k
        self.selected_indices_ = ranked_indices[:self.k]

        self.support_mask_ = np.zeros(
            X.shape[1],
            dtype=bool
        )
        self.support_mask_[self.selected_indices_] = True

        self.n_components_retained_ = self.pca_.n_components_
        self.cumulative_variance_ = (
            self.pca_.explained_variance_ratio_.sum()
        )

        return self

    def transform(self, X):
        # Return selected ORIGINAL gene-expression columns
        if hasattr(X, "iloc"):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

    def get_support(self):
        return self.support_mask_
