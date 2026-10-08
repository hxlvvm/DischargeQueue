"""Candidate models. Each returns an sklearn pipeline that takes the output of build_features."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .features import UTILISATION, make_preprocessor


def utilisation_lr():
    """Learned baseline: logistic regression on prior-year outpatient, emergency and inpatient visit
    counts only. (LACE would need a Charlson comorbidity score, which three diagnosis codes
    cannot supply.)"""
    pre = ColumnTransformer([("util", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                              UTILISATION)])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


def full_lr():
    """All discharge-time features, L2-regularised logistic regression."""
    return make_pipeline(make_preprocessor(scale=True), LogisticRegression(C=0.1, max_iter=2000))


def full_gbm():
    """All discharge-time features, histogram gradient boosting with early stopping."""
    return make_pipeline(make_preprocessor(scale=False),
                         HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31,
                                                        min_samples_leaf=50, l2_regularization=1.0,
                                                        early_stopping=True, validation_fraction=0.1,
                                                        random_state=SEED))


def prior_inpatient_rule(X, seed=SEED):
    """Rule baseline: rank by prior-year inpatient visits, then emergency visits; remaining ties are
    broken by a seeded random draw so that top-k membership is reproducible."""
    jitter = np.random.default_rng(seed).random(len(X)) * 1e-3
    return X["number_inpatient"].to_numpy() + 1e-2 * X["number_emergency"].clip(upper=9).to_numpy() + jitter
