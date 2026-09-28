from __future__ import annotations


import hashlib
import importlib
import importlib.metadata
import platform
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


STAGE4_DIRECT_FEATURES: tuple[str, ...] = (
    "WBC", "NEUT#", "NEUT%", "MONO#", "MONO%", "PLT", "RDW-CV", "CRP", "FIB", "LDH", "PCT",
    "LYMPH#", "LYMPH%", "EO#", "EO%", "BASO#", "BASO%", "HGB", "ALB", "TP", "GLB",
    "PA", "GLU", "TG", "TC", "HDL-C", "LDL-C", "UA", "MCH", "MCHC", "Cl", "Na",
)
STAGE4_COMPOSITE_FEATURES: tuple[str, ...] = ("SIRI", "LMR", "GAR", "PNI", "HALP")
STAGE4_FEATURES: tuple[str, ...] = STAGE4_DIRECT_FEATURES + STAGE4_COMPOSITE_FEATURES

MASTER_SEED = 20260902
SEED_NAMESPACE = "SCLC_NSCLC|model_factory_v2_23design"
MODEL_FACTORY_VERSION = "stage6_model_factory_v2_23design"
ROTATION_FOREST_IMPLEMENTATION_VERSION = "rotation_forest_custom_sklearn_v1"

MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "gam",
    "knn",
    "rbf_svm",
    "gaussian_nb",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gbdt",
    "xgboost",
    "lightgbm",
    "adaboost",
    "rotation_forest",
    "mlp",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "gam": "GAM",
    "knn": "KNN",
    "rbf_svm": "RBF-SVM",
    "gaussian_nb": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gbdt": "GBDT",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "adaboost": "AdaBoost",
    "rotation_forest": "Rotation Forest",
    "mlp": "MLP",
}

SCALER_POLICY: dict[str, str] = {
    "logistic_regression": "standard_scaler_training_fold",
    "gam": "passthrough",
    "knn": "standard_scaler_training_fold",
    "rbf_svm": "standard_scaler_training_fold",
    "gaussian_nb": "standard_scaler_training_fold",
    "decision_tree": "passthrough",
    "random_forest": "passthrough",
    "extra_trees": "passthrough",
    "gbdt": "passthrough",
    "xgboost": "passthrough",
    "lightgbm": "passthrough",
    "adaboost": "passthrough",
    "rotation_forest": "passthrough",
    "mlp": "standard_scaler_training_fold",
}

_NAME_ALIASES: dict[str, str] = {
    "lr": "logistic_regression",
    "logistic": "logistic_regression",
    "logistic regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "gam": "gam",
    "logisticgam": "gam",
    "knn": "knn",
    "k-nearest neighbors": "knn",
    "rbf-svm": "rbf_svm",
    "rbf_svm": "rbf_svm",
    "svm": "rbf_svm",
    "gaussian nb": "gaussian_nb",
    "gaussian naive bayes": "gaussian_nb",
    "gaussian_nb": "gaussian_nb",
    "decision tree": "decision_tree",
    "decision_tree": "decision_tree",
    "random forest": "random_forest",
    "random_forest": "random_forest",
    "extra trees": "extra_trees",
    "extra_trees": "extra_trees",
    "gbdt": "gbdt",
    "gradient boosting": "gbdt",
    "gradient_boosting": "gbdt",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "adaboost": "adaboost",
    "ada boost": "adaboost",
    "rotation forest": "rotation_forest",
    "rotation_forest": "rotation_forest",
    "rotationforest": "rotation_forest",
    "mlp": "mlp",
    "multilayer perceptron": "mlp",
}


DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000,
        "class_weight": None,
    },
    "gam": {
        "n_splines": 10, "spline_order": 3, "lam": 0.6, "max_iter": 100,
        "tol": 1e-4, "fit_intercept": True, "internal_standardize": True,
    },
    "knn": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2},
    "rbf_svm": {
        "C": 1.0, "gamma": "scale", "kernel": "rbf", "class_weight": None,
        "probability": False, "max_iter": -1,
    },
    "gaussian_nb": {"var_smoothing": 1e-9},
    "decision_tree": {
        "criterion": "gini", "max_depth": None, "min_samples_leaf": 1,
        "class_weight": None,
    },
    "random_forest": {
        "n_estimators": 200, "criterion": "gini", "max_depth": None,
        "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None,
        "n_jobs": 1,
    },
    "extra_trees": {
        "n_estimators": 200, "criterion": "gini", "max_depth": None,
        "min_samples_leaf": 1, "max_features": 1.0, "class_weight": None,
        "n_jobs": 1,
    },
    "gbdt": {
        "n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
        "min_samples_leaf": 1, "subsample": 1.0,
    },
    "xgboost": {
        "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9, "objective": "binary:logistic",
        "eval_metric": "logloss", "n_jobs": 1, "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200, "num_leaves": 15, "max_depth": -1,
        "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9,
        "objective": "binary", "n_jobs": 1, "verbosity": -1,
    },
    "adaboost": {"n_estimators": 100, "learning_rate": 0.05, "algorithm": "SAMME"},
    "rotation_forest": {
        "n_estimators": 100, "n_subsets": 3, "criterion": "gini",
        "max_depth": None, "min_samples_leaf": 1,
    },
    "mlp": {
        "hidden_layer_sizes": [32], "activation": "relu", "solver": "adam",
        "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 500,
        "early_stopping": False, "batch_size": "auto",
    },
}


def derive_seed(*parts: object) -> int:

    payload = "|".join([SEED_NAMESPACE, str(MASTER_SEED), *(str(x) for x in parts)])
    return 1 + int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % 2_147_483_646


def canonical_model_name(name: str) -> str:

    if not isinstance(name, str):
        raise TypeError("model name must be a string")
    original = name.strip().lower()
    if original in _NAME_ALIASES:
        return _NAME_ALIASES[original]
    key = " ".join(original.replace("_", " ").split())

    key = key.replace("–", "-").replace("—", "-")
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    if key.replace("-", " ") in _NAME_ALIASES:
        return _NAME_ALIASES[key.replace("-", " ")]
    raise ValueError(f"unknown base model {name!r}; expected one of {MODEL_NAMES}")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_versions() -> dict[str, str | None]:

    return {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "scipy": _version("scipy"),
        "joblib": _version("joblib"),
        "pygam": _version("pygam"),
        "xgboost": _version("xgboost"),
        "lightgbm": _version("lightgbm"),
        "sktime": _version("sktime"),
    }


def environment_info() -> dict[str, Any]:

    return {
        "model_factory_version": MODEL_FACTORY_VERSION,
        "rotation_forest_implementation": ROTATION_FOREST_IMPLEMENTATION_VERSION,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }


def _as_stage4_frame(X: Any) -> pd.DataFrame:

    if isinstance(X, pd.DataFrame):
        missing = [c for c in STAGE4_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Stage-4 input is missing required columns: {missing}")
        return X.loc[:, list(STAGE4_FEATURES)].copy()
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] != len(STAGE4_FEATURES):
        raise ValueError(f"Stage-4 array must have shape (n, {len(STAGE4_FEATURES)})")
    return pd.DataFrame(array, columns=STAGE4_FEATURES)


class Stage4Selector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        selected_direct: Sequence[str] | None = None,
        include_composites: bool = True,
        scope_id: str | None = None,
    ) -> None:
        self.selected_direct = tuple(selected_direct) if selected_direct is not None else None
        self.include_composites = include_composites
        self.scope_id = scope_id

    def fit(self, X: Any, y: Any = None) -> "Stage4Selector":
        frame = _as_stage4_frame(X)
        if self.selected_direct is None:
            direct = list(STAGE4_DIRECT_FEATURES)
        else:
            direct = list(self.selected_direct)
            if len(set(direct)) != len(direct):
                raise ValueError("selected_direct contains duplicate feature names")
            unknown = [c for c in direct if c not in STAGE4_DIRECT_FEATURES]
            if unknown:
                raise ValueError(f"selector may only select direct Stage-4 features: {unknown}")
        if not self.include_composites:
            raise ValueError("composite features are protocol-forced and cannot be removed")
        self.selected_direct_ = tuple(direct)
        self.selected_features_ = tuple(direct + list(STAGE4_COMPOSITE_FEATURES))
        self.n_features_in_ = frame.shape[1]
        self.n_features_out_ = len(self.selected_features_)
        self.feature_names_in_ = np.asarray(STAGE4_FEATURES, dtype=object)
        self.fit_n_samples_ = int(frame.shape[0])
        self.fit_scope_id_ = self.scope_id
        self.forced_composites_ = tuple(STAGE4_COMPOSITE_FEATURES)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Stage4Selector must be fit before transform")
        frame = _as_stage4_frame(X)
        return frame.loc[:, list(self.selected_features_)].copy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Stage4Selector must be fit before get_feature_names_out")
        return np.asarray(self.selected_features_, dtype=object)


class DecisionFunctionOnlySVC(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        class_weight: Any = None,
        probability: bool = False,
        max_iter: int = -1,
        random_state: int | None = None,
    ) -> None:
        if probability:
            raise ValueError("DecisionFunctionOnlySVC forbids probability=True; RBF-SVM reports a decision score only")
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.class_weight = class_weight
        self.probability = probability
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "DecisionFunctionOnlySVC":
        if self.probability:
            raise ValueError("probability=True is prohibited for the project SVM")
        self._svc = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            class_weight=self.class_weight,
            probability=False,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._svc.fit(X, y)
        self.classes_ = np.asarray(self._svc.classes_)
        self.n_features_in_ = getattr(self._svc, "n_features_in_", np.asarray(X).shape[1])
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_svc"):
            raise RuntimeError("DecisionFunctionOnlySVC must be fit before decision_function")
        return self._svc.decision_function(X)

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_svc"):
            raise RuntimeError("DecisionFunctionOnlySVC must be fit before predict")
        return self._svc.predict(X)


class _UnavailableOptionalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, dependency: str, model_name: str, params: Mapping[str, Any] | None = None) -> None:
        self.dependency = dependency
        self.model_name = model_name
        self.params = dict(params or {})

    def fit(self, X: Any, y: Any) -> "_UnavailableOptionalClassifier":
        raise ImportError(
            f"{self.model_name} requires optional dependency {self.dependency!r}; "
            "install and lock the package before formal training"
        )

    def predict_proba(self, X: Any) -> np.ndarray:


        self.fit(X, np.zeros(len(np.asarray(X))))
        raise AssertionError("unreachable")

    def predict(self, X: Any) -> np.ndarray:
        self.fit(X, np.zeros(len(np.asarray(X))))
        raise AssertionError("unreachable")


class PygamLogisticClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_splines: int = 10,
        spline_order: int = 3,
        lam: float = 0.6,
        max_iter: int = 100,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        internal_standardize: bool = True,
    ) -> None:
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.internal_standardize = internal_standardize

    def fit(self, X: Any, y: Any) -> "PygamLogisticClassifier":
        try:
            from pygam import LogisticGAM, s
        except ImportError as exc:
            raise ImportError("GAM requires pygam; install and lock pygam before formal training") from exc
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("GAM input must be a 2-D matrix")
        if not np.isfinite(array).all():
            raise ValueError("GAM input contains non-finite values")


        if bool(self.internal_standardize):
            self._internal_mean_ = np.mean(array, axis=0)
            self._internal_scale_ = np.std(array, axis=0)
            self._internal_scale_[self._internal_scale_ <= np.finfo(float).eps] = 1.0
            fit_array = (array - self._internal_mean_) / self._internal_scale_
        else:
            self._internal_mean_ = np.zeros(array.shape[1], dtype=float)
            self._internal_scale_ = np.ones(array.shape[1], dtype=float)
            fit_array = array
        terms = s(0, n_splines=self.n_splines, spline_order=self.spline_order)
        for index in range(1, array.shape[1]):
            terms += s(index, n_splines=self.n_splines, spline_order=self.spline_order)
        self._fit_lam_ = float(self.lam)
        self._fit_retry_count_ = 0


        gam = LogisticGAM(
            terms=terms,
            lam=float(self.lam),
            max_iter=max(int(self.max_iter), 200),
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            callbacks=[],
        )
        try:
            gam.fit(fit_array, np.asarray(y))
        except Exception as exc:
            if hasattr(self, "_gam"):
                del self._gam
            raise RuntimeError(
                f"GAM fitting failed at the frozen lam={self.lam}; "
                "automatic penalty changes are prohibited by the manuscript protocol"
            ) from exc
        self._gam = gam
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = array.shape[1]
        return self

    def _check_fit(self) -> None:
        if not hasattr(self, "_gam"):
            raise RuntimeError("PygamLogisticClassifier must be fit before prediction")

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fit()
        array = np.asarray(X, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("GAM input contains non-finite values")
        array = (array - self._internal_mean_) / self._internal_scale_
        p1 = np.asarray(self._gam.predict_proba(array), dtype=float).reshape(-1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.2076).astype(int)

    def decision_function(self, X: Any) -> np.ndarray:
        p1 = np.clip(self.predict_proba(X)[:, 1], 1e-15, 1.0 - 1e-15)
        return np.log(p1 / (1.0 - p1))


@dataclass
class _RotationBlock:
    indices: np.ndarray
    pca: PCA
    output_width: int


class RotationForestClassifier(BaseEstimator, ClassifierMixin):

    implementation_version: ClassVar[str] = ROTATION_FOREST_IMPLEMENTATION_VERSION

    def __init__(
        self,
        n_estimators: int = 100,
        n_subsets: int = 3,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_subsets = n_subsets
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _rotate(self, X: np.ndarray, blocks: Sequence[_RotationBlock]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for block in blocks:
            transformed = np.asarray(block.pca.transform(X[:, block.indices]), dtype=float)
            if transformed.shape[1] < block.output_width:
                transformed = np.pad(
                    transformed,
                    ((0, 0), (0, block.output_width - transformed.shape[1])),
                    mode="constant",
                )
            parts.append(transformed[:, : block.output_width])
        return np.concatenate(parts, axis=1)

    def fit(self, X: Any, y: Any) -> "RotationForestClassifier":
        array = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if array.ndim != 2:
            raise ValueError("RotationForest input must be a 2-D matrix")
        if array.shape[0] != len(labels):
            raise ValueError("RotationForest X/y row counts differ")
        if array.shape[0] < 2 or len(np.unique(labels)) < 2:
            raise ValueError("RotationForest requires at least two training classes")
        if self.n_estimators < 1 or self.n_subsets < 1:
            raise ValueError("n_estimators and n_subsets must be positive")
        self.classes_ = np.unique(labels)
        self.n_features_in_ = array.shape[1]
        rng = np.random.RandomState(self.random_state if self.random_state is not None else derive_seed("rotation"))
        self.rotations_: list[tuple[_RotationBlock, ...]] = []
        self.estimators_: list[DecisionTreeClassifier] = []
        for tree_index in range(self.n_estimators):
            permutation = rng.permutation(self.n_features_in_)
            groups = [np.asarray(x, dtype=int) for x in np.array_split(permutation, min(self.n_subsets, self.n_features_in_)) if len(x)]
            blocks: list[_RotationBlock] = []
            for group_index, indices in enumerate(groups):


                n_components = min(len(indices), array.shape[0])
                pca = PCA(n_components=n_components, svd_solver="full")
                pca.fit(array[:, indices])
                blocks.append(_RotationBlock(indices=indices, pca=pca, output_width=len(indices)))
            rotated = self._rotate(array, blocks)
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.randint(1, 2_147_483_646)),
            )
            tree.fit(rotated, labels)
            self.rotations_.append(tuple(blocks))
            self.estimators_.append(tree)
        return self

    def _check_fit(self) -> None:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("RotationForestClassifier must be fit before prediction")

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fit()
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(f"RotationForest input must have {self.n_features_in_} columns")
        total = np.zeros((array.shape[0], len(self.classes_)), dtype=float)
        for tree, blocks in zip(self.estimators_, self.rotations_):
            tree_prob = tree.predict_proba(self._rotate(array, blocks))
            aligned = np.zeros_like(total)
            for pos, cls in enumerate(tree.classes_):
                global_pos = int(np.flatnonzero(self.classes_ == cls)[0])
                aligned[:, global_pos] = tree_prob[:, pos]
            total += aligned
        total /= float(len(self.estimators_))
        return total

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.shape[1] != 2:
            raise ValueError("decision_function is defined only for binary Rotation Forest")
        return proba[:, 1] - proba[:, 0]


def _make_estimator(model_name: str, seed: int, params: Mapping[str, Any]) -> BaseEstimator:

    name = canonical_model_name(model_name)
    kw = dict(params)
    estimator_seed = kw.pop("random_state", seed)
    if name == "logistic_regression":
        return LogisticRegression(random_state=estimator_seed, **kw)
    if name == "gam":
        return PygamLogisticClassifier(**kw)
    if name == "knn":
        return KNeighborsClassifier(**kw)
    if name == "rbf_svm":
        return DecisionFunctionOnlySVC(random_state=estimator_seed, **kw)
    if name == "gaussian_nb":
        return GaussianNB(**kw)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=estimator_seed, **kw)
    if name == "random_forest":
        return RandomForestClassifier(random_state=estimator_seed, **kw)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=estimator_seed, **kw)
    if name == "gbdt":
        return GradientBoostingClassifier(random_state=estimator_seed, **kw)
    if name == "xgboost":
        try:
            module = importlib.import_module("xgboost")
        except ImportError:
            return _UnavailableOptionalClassifier("xgboost", "XGBoost", kw)
        return module.XGBClassifier(random_state=estimator_seed, **kw)
    if name == "lightgbm":
        try:
            module = importlib.import_module("lightgbm")
        except ImportError:
            return _UnavailableOptionalClassifier("lightgbm", "LightGBM", kw)
        return module.LGBMClassifier(random_state=estimator_seed, **kw)
    if name == "adaboost":
        base_tree = DecisionTreeClassifier(max_depth=1, random_state=derive_seed("adaboost_base", estimator_seed))


        try:
            return AdaBoostClassifier(estimator=base_tree, random_state=estimator_seed, **kw)
        except TypeError:
            return AdaBoostClassifier(base_estimator=base_tree, random_state=estimator_seed, **kw)
    if name == "rotation_forest":
        return RotationForestClassifier(random_state=estimator_seed, **kw)
    if name == "mlp":
        return MLPClassifier(random_state=estimator_seed, **kw)
    raise AssertionError(f"unhandled model {name}")


def _clone_selector(selector: Stage4Selector | BaseEstimator | None) -> BaseEstimator:
    if selector is None:
        return Stage4Selector()
    try:
        return clone(selector)
    except Exception as exc:
        raise TypeError("selector must be a sklearn-compatible transformer") from exc


def build_model(
    model_name: str,
    *,
    seed: int | None = None,
    selector: Stage4Selector | BaseEstimator | None = None,
    selected_direct: Sequence[str] | None = None,
    scope_id: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> Pipeline:

    name = canonical_model_name(model_name)
    model_seed = int(MASTER_SEED if seed is None else seed)
    model_params = deepcopy(DEFAULT_MODEL_PARAMS[name])
    if params:
        model_params.update(dict(params))
    if selected_direct is not None:
        if selector is not None:
            raise ValueError("pass either selector or selected_direct, not both")
        selector_obj: BaseEstimator = Stage4Selector(selected_direct=selected_direct, scope_id=scope_id)
    else:
        selector_obj = _clone_selector(selector)
        if isinstance(selector_obj, Stage4Selector) and scope_id is not None:
            selector_obj.scope_id = scope_id
    estimator = _make_estimator(name, model_seed, model_params)
    steps: list[tuple[str, Any]] = [("selector", selector_obj)]
    if SCALER_POLICY[name] == "standard_scaler_training_fold":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps, memory=None)


make_model = build_model
make_base_model = build_model
model_factory = build_model


__all__ = [
    "MASTER_SEED",
    "MODEL_FACTORY_VERSION",
    "ROTATION_FOREST_IMPLEMENTATION_VERSION",
    "STAGE4_DIRECT_FEATURES",
    "STAGE4_COMPOSITE_FEATURES",
    "STAGE4_FEATURES",
    "MODEL_NAMES",
    "MODEL_DISPLAY_NAMES",
    "SCALER_POLICY",
    "DEFAULT_MODEL_PARAMS",
    "derive_seed",
    "canonical_model_name",
    "dependency_versions",
    "environment_info",
    "Stage4Selector",
    "DecisionFunctionOnlySVC",
    "PygamLogisticClassifier",
    "RotationForestClassifier",
    "build_model",
    "make_model",
    "make_base_model",
    "model_factory",
]
