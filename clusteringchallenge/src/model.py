
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PipelineWrapper:
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []

        # Placeholder for fitted transformers & model
        self.preprocessor: Pipeline = None
        self.reducer: PCA = None
        self.clusterer: GaussianMixture = None
        self.fitted = False


    def _init_preprocessor(self, X: pd.DataFrame):
        """
        Build a ColumnTransformer that:
          - scales numerical features
          - ordinal-encodes categorical features
        """
        # On first call, infer which columns are numeric vs categorical
        # Here we assume pandas DataFrame.
        if isinstance(X, pd.DataFrame):
            # Split by dtype: numeric vs categorical
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()
            # exclude target if present (but in clustering no target)
            cat_cols = [c for c in X.columns if c not in num_cols]
            self.numerical_features = num_cols
            self.categorical_features = cat_cols

        # Build transformer
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numerical_features),
            ("cat", OrdinalEncoder(),   self.categorical_features)
        ], remainder="drop")

    def fit(self, X, y=None):
        """
        :param X: pandas DataFrame or array-like of shape (n_samples, n_features)
        :param y: ignored (for compatibility)
        """
        # Convert to DataFrame for easy column handling, if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize preprocessor on the first fit
        if self.preprocessor is None:
            self._init_preprocessor(X)

        # Transform features
        X_prep = self.preprocessor.fit_transform(X)

        # Dimensionality reduction: keep up to  min(n_samples, n_features, 10) PCs
        n_samples, n_feats = X_prep.shape
        n_comp = min(10, n_feats, n_samples)
        self.reducer = PCA(n_components=n_comp, random_state=42)
        X_red = self.reducer.fit_transform(X_prep)

        # Select best num_clusters via BIC over GaussianMixture(2..12)
        best_bic = np.inf
        best_gmm = None
        for k in range(2, 13):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42
            )
            gmm.fit(X_red)
            bic = gmm.bic(X_red)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        self.clusterer = best_gmm
        self.fitted = True
        return self

    def predict(self, X):
        """
        :param X: pandas DataFrame or array-like of shape (n_samples, n_features)
        :return: numpy array of cluster labels (integers from 0 to k-1)
        """
        if not self.fitted:
            raise RuntimeError("PipelineWrapper: fit() must be called before predict().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.numerical_features + self.categorical_features)

        X_prep = self.preprocessor.transform(X)
        X_red = self.reducer.transform(X_prep)
        labels = self.clusterer.predict(X_red)
        return np.asarray(labels)
