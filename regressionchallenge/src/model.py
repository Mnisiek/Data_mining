import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


class PipelineWrapper:
    def __init__(self):
        self.categorical_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
        self.numerical_features = ["temp", "atemp", "hum", "windspeed"]

        # Defining the preprocessor: OneHotEncoder for categorical variables, StandardScaler for numerical variables
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("numerical", StandardScaler(), self.numerical_features)
            ],
            remainder="drop"
        )

        # Model regressor - MultiOutput
        self.regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", self.regressor)
        ])

        self.fitted = False


    def _preprocess_X(self, X):
        """
        If X is a DataFrame, try to perform data cleaning. Otherwise assume that X is already prepared a numpy array.
        """

        if isinstance(X, pd.DataFrame):
            # print(f"[DEBUG] Original DataFrame columns: {list(X.columns)}", file=sys.stderr)
            drop_cols = [col for col in ["instant", "dteday", "casual", "registered", "cnt"] if col in X.columns]
            if drop_cols:
                X = X.drop(columns=drop_cols)
                # print(f"[DEBUG] Dropped columns: {drop_cols}", file=sys.stderr)
            # missing_categorical = [col for col in self.categorical_features if col not in X.columns]
            # missing_numerical = [col for col in self.numerical_features if col not in X.columns]
            # if missing_categorical or missing_numerical:
                # print(
                    # f"[WARNING] Missing expected columns. Categorical missing: {missing_categorical}, "
                    # f"Numeric missing: {missing_numerical}", file=sys.stderr)
            return X
        else:
            return X


    def fit(self, X, y):
        """
        :param X: pandas DataFrame or numpy array with features
        :param y: numpy array shaped (n_samples, 2)
        """

        X_prepared = self._preprocess_X(X)

        self.pipeline.fit(X_prepared, y)
        self.fitted = True
        # print("[INFO] Model fitted successfully.", file=sys.stderr)

    def predict(self, X):
        """
        :param X: pandas DataFrame or numpy array with features
        :return: tuple of two numpy array: (predictions_for_casual, predictions_for_registered)
        """
        if not self.fitted:
            # print("[ERROR] Model is not fitted yet. Call fit first.", file=sys.stderr)
            return np.array([]), np.array([])

        X_prepared = self._preprocess_X(X)

        preds = self.pipeline.predict(X_prepared)

        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 2)
        casual_pred = preds[:, 0]
        registered_pred = preds[:, 1]
        return casual_pred, registered_pred
