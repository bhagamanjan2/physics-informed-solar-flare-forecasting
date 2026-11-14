"""
Machine learning models and pipelines for solar flare classification.
Owned by Agent A3 (Split and Models).

Models:
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Decision Tree
  4. Random Forest
  5. Support Vector Machine (SVM RBF)
  6. Gaussian Naive Bayes
  7. Gradient Boosting
Baselines:
  - All-Quiet baseline (always predicts 0)
  - Flux-Threshold baseline (predicts 1 if flux exceeds threshold)
  - Recent-Flare baseline (predicts 1 if flare occurred in lookback)
"""

from typing import Dict, Any, List, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV

from solarflare.split import blocked_time_series_cv


class AllQuietBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline that always predicts no flare (class 0).
    Represents the naive zero-flare hypothesis.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        prob_0 = np.ones(n, dtype=float)
        prob_1 = np.zeros(n, dtype=float)
        return np.column_stack([prob_0, prob_1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class FluxThresholdBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline that predicts flare if flux or net flux exceeds a threshold.
    Default threshold is -6.0 in log10 space (corresponding to 1.0e-6 W/m^2 C-class).
    """
    def __init__(self, threshold: float = -6.0):
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _extract_flux(self, X):
        if isinstance(X, pd.DataFrame):
            for col in ["g0_log_net_flux", "net_flux", "flux"]:
                if col in X.columns:
                    return X[col].values
            return X.iloc[:, 0].values
        elif isinstance(X, np.ndarray):
            return X[:, 0]
        else:
            return np.array(X)[:, 0]

    def predict_proba(self, X):
        flux = self._extract_flux(X)
        prob_1 = (flux >= self.threshold).astype(float)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RecentFlareBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline that predicts flare if a flare ended or started recently
    (e.g. within window_min minutes).
    """
    def __init__(self, window_min: float = 180.0):
        self.window_min = window_min
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _extract_history(self, X):
        if isinstance(X, pd.DataFrame):
            if "g1_dt_last_flare" in X.columns:
                return X["g1_dt_last_flare"].values
            elif "g1_n_flares_6h" in X.columns:
                return (X["g1_n_flares_6h"].values > 0).astype(float)
        return np.zeros(len(X))

    def predict_proba(self, X):
        dt = self._extract_history(X)
        if isinstance(X, pd.DataFrame) and "g1_dt_last_flare" in X.columns:
            prob_1 = ((dt >= 0) & (dt <= self.window_min)).astype(float)
        else:
            prob_1 = (dt > 0).astype(float)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# Alias for backward compatibility
PersistenceBaseline = FluxThresholdBaseline


def build_pipeline(
    model_name: str, class_weight: str = "balanced", random_state: int = 42
) -> Pipeline:
    """
    Build scikit-learn Pipeline with appropriate scaler and model.

    Parameters
    ----------
    model_name : str
        Name or alias of the model.
    class_weight : str
        Class weight setting ('balanced' or None).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Configured scikit-learn pipeline.
    """
    m = model_name.lower().replace("-", "_")

    if m in ["lr", "logistic_regression"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)),
        ])

    elif m in ["knn", "kneighbors"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        ])

    elif m in ["dt", "decision_tree"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(class_weight=class_weight, random_state=random_state, max_depth=6)),
        ])

    elif m in ["rf", "random_forest"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100, class_weight=class_weight, random_state=random_state, max_depth=8, n_jobs=-1
            )),
        ])

    elif m in ["svm", "svc", "svm_rbf"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", probability=True, class_weight=class_weight, random_state=random_state, max_iter=2000
            )),
        ])

    elif m in ["nb", "naive_bayes", "gaussian_nb"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GaussianNB()),
        ])

    elif m in ["gb", "gradient_boosting"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=60, random_state=random_state, max_depth=4)),
        ])

    elif m in ["all_quiet", "quiet"]:
        return Pipeline([("clf", AllQuietBaseline())])

    elif m in ["flux_threshold", "threshold", "baseline"]:
        return Pipeline([("clf", FluxThresholdBaseline())])

    elif m in ["recent_flare", "persistence"]:
        return Pipeline([("clf", RecentFlareBaseline())])

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_param_distributions(model_name: str) -> Dict[str, Any]:
    """
    Return hyperparameter search distribution for RandomizedSearchCV.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    dict
        Dictionary of parameter search grids for scikit-learn.
    """
    m = model_name.lower().replace("-", "_")

    if m in ["lr", "logistic_regression"]:
        return {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    elif m in ["knn", "kneighbors"]:
        return {"clf__n_neighbors": [5, 15, 25]}

    elif m in ["dt", "decision_tree"]:
        return {
            "clf__max_depth": [4, 6, 8],
            "clf__min_samples_split": [10, 50],
        }

    elif m in ["rf", "random_forest"]:
        return {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [6, 10],
        }

    elif m in ["svm", "svc", "svm_rbf"]:
        return {
            "clf__C": [0.1, 1.0],
            "clf__gamma": ["scale", "auto"],
        }

    elif m in ["nb", "naive_bayes", "gaussian_nb"]:
        return {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]}

    elif m in ["gb", "gradient_boosting"]:
        return {
            "clf__n_estimators": [50, 60],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.05, 0.1],
        }

    else:
        return {}


def train_and_tune(
    pipeline: Pipeline,
    param_dist: dict,
    X_train,
    y_train,
    cv_splits=None,
    fast_mode: bool = False,
    n_iter: int = 4,
    random_state: int = 42,
) -> Pipeline:
    """
    Tune and fit model using blocked CV.
    In fast_mode: subsamples 30k-40k rows spread across training for slow models.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted scikit-learn pipeline.
    param_dist : dict
        Hyperparameter distribution dictionary.
    X_train : pd.DataFrame or np.ndarray
        Feature matrix for training.
    y_train : pd.Series or np.ndarray
        Binary target vector.
    cv_splits : list or generator, optional
        Precomputed blocked cross-validation splits.
    fast_mode : bool
        If True, use lightweight parameter search and subsample KNN/SVM.
    n_iter : int
        Number of iterations for RandomizedSearchCV.
    random_state : int
        Random seed.

    Returns
    -------
    Pipeline
        Fitted best estimator.
    """
    clf_step = pipeline.named_steps.get("clf", None)
    is_slow = isinstance(clf_step, (KNeighborsClassifier, SVC, GradientBoostingClassifier, RandomForestClassifier))
    n_train = len(X_train)

    # Fast mode: subsample 30k-40k rows spread over training period for slow models
    if fast_mode and is_slow and n_train > 35000:
        sub_idx = np.linspace(0, n_train - 1, 35000, dtype=int)
        X_fit = X_train.iloc[sub_idx] if hasattr(X_train, "iloc") else X_train[sub_idx]
        y_fit = y_train.iloc[sub_idx] if hasattr(y_train, "iloc") else y_train[sub_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # In fast mode, fit directly with default robust hyperparameters for SVM to avoid 20x overhead
        if param_dist and cv_splits is not None and not (fast_mode and isinstance(clf_step, SVC)):
            actual_iter = min(n_iter if fast_mode else (n_iter * 2), len(param_dist) * 2, 2 if is_slow else 3)
            
            if len(X_fit) != n_train:
                n_folds = len(cv_splits) if hasattr(cv_splits, "__len__") else 5
                fit_cv = blocked_time_series_cv(X_fit, n_splits=n_folds)
            else:
                fit_cv = cv_splits

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=actual_iter,
                cv=fit_cv,
                scoring="roc_auc",
                random_state=random_state,
                n_jobs=1 if is_slow else -1,
                refit=True,
                error_score=0.5,
            )
            search.fit(X_fit, y_fit)
            best_model = search.best_estimator_
            best_model.best_params_ = search.best_params_
            best_model.cv_results_ = search.cv_results_
            return best_model
        else:
            pipeline.fit(X_fit, y_fit)
            pipeline.best_params_ = {}
            return pipeline
