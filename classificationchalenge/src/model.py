import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class PipelineWrapper:
    def __init__(self):
        # Defining pipeline of TF-IDF
        base_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1,2),
                max_df=0.9,
                min_df=5
            )),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="saga",
                max_iter=2000,
                random_state=42,
                n_jobs=-1
            ))
        ])
        self.best_pipeline = None

        # Hyperparameter grid for CV
        self.param_grid = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_df": [0.8, 0.9],
            "clf__C": [0.1, 1.0, 5.0]
        }

        self.base_pipe = base_pipe
        self.grid: GridSearchCV = None
        self.fitted = False


    def _prepare_texts(self, X):
        """
        :param X: DataFrame with a 'text' column or array-like of raw strings
        :return: a Python list of strings (length = n_samples)
        """
        if isinstance(X, pd.DataFrame):
            if "text" not in X.columns:
                raise ValueError("DataFrame must contain a 'text' column")
            # ensure dtype=str
            return X["text"].astype(str).tolist()
        else:
            # assume array-like of raw strings
            return [str(x) for x in X]


    def fit(self, X, y):
        """
        :param X: DataFrame with 'text' or array-like of strings
        :param y: array-like of labels (0,1,2)
        """
        texts = self._prepare_texts(X)

        # set up and run grid search
        self.grid = GridSearchCV(
            estimator=self.base_pipe,
            param_grid=self.param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            error_score='raise',
            verbose=0
        )
        self.grid.fit(texts, y)

        # select the best pipeline
        self.best_pipeline = self.grid.best_estimator_
        self.fitted = True


    def predict(self, X):
        """
        :param X: DataFrame with 'text' or array-like of strings
        :return: np.array of predicted labels
        """

        texts = self._prepare_texts(X)
        preds = self.best_pipeline.predict(texts)
        return np.asarray(preds)
