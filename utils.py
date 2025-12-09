import pandas as pd
import lir
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import math

from typing import Callable, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Sized
from lir import check_misleading_Inf_negInf, Xy_to_Xn

from models.CatBoost import CatBoost
from models.DecisionTree import DecisionTree
from models.RandomForest import RandomForest
from models.XGBoost import XGBoost

LOG = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

def load_data(cfg):
    data = pd.read_csv(cfg.eval.data_path)
    return data

def split_train_select(cfg, data, sel_set):
    # Combine supplementary data with incomplete data
    if cfg.data.name == 'NFI':
        pp_map = {'pp11': 'pp03', 'pp12': 'pp05', 'pp15': 'pp04'} # just swapped pp05 and pp04 (was pp12:pp04 and pp15:pp05)
        data['META_test_subject'] = data['META_test_subject'].map(lambda x: pp_map.get(x, x))
    
    if not cfg.eval.is_multiclass:
        activity_1, activity_2 = cfg.eval.activity_pair
        assert activity_1 != activity_2, f"Null and alternative activities must be different ({activity_1})"
        
        # Select only rows with the specified activities (H1, H0)
        acts_to_use = list((activity_1, activity_2))
        data = data[data['META_label_activity'].isin(acts_to_use)].copy()
        data["label"] = data["META_label_activity"].map({activity_1: 1, activity_2: 0})
    else:
        cluster_map = cfg.eval.expert_cluster_assignment
        data["label"] = data["META_label_activity"].map(cluster_map)
        n_clusters = len(set(cluster_map.values()))
        if cfg.eval.expert_cluster_choices is not None:
            print(f"Selecting only expert clusters: {cfg.eval.expert_cluster_choices}")
            # Filter data to include only rows with activities in the selected expert clusters
            desired_clusters = [cfg.eval.expert_cluster_map[choice] for choice in cfg.eval.expert_cluster_choices]
            n_clusters = len(desired_clusters)
            data = data[data["label"].isin(desired_clusters)]

    # Select specified phone types
    if cfg.eval.phone_types is not None:
        data = data[data['META_telephone_type'].isin(cfg.eval.phone_types)].copy()
    # Select specified carry locations
    if cfg.eval.carry_locations is not None:
        data = data[data['META_carrying_location'].isin(cfg.eval.carry_locations)].copy()

    # Split data into train, sel on basis of pre selected subjects
    data['META_test_subject'] = data['META_test_subject'].str.extract('(\d+)').astype(int)
    sel_data = data[data['META_test_subject'].isin(sel_set)]
    train_data = data[~data['META_test_subject'].isin(sel_set)]

    # Drop columns not to be used
    cols_to_use = [col for col in cfg.data.vars_to_use if col in train_data.columns]    
    sel_vars = sel_data.filter(items=cols_to_use, axis=1)
    train_vars = train_data.filter(items=cols_to_use, axis=1)
    sel_labels = sel_data['label']
    train_labels = train_data['label']
    sel_pp = sel_data['META_test_subject']
    train_pp = train_data['META_test_subject']
    sel_phone = sel_data['META_telephone_type']
    train_phone = train_data['META_telephone_type']
    sel_carryloc = sel_data['META_carrying_location']
    train_carryloc = train_data['META_carrying_location']

    # Normalise columns if specified
    if cfg.eval.normalise:
        normalisable_cols = [col for col in cfg.data.normalisable_vars if col in train_vars.columns]
        # Training and selection data normalised using training data mean and std
        train_mean = train_vars[normalisable_cols].mean()
        train_std = train_vars[normalisable_cols].std()
        train_vars[normalisable_cols] = (train_vars[normalisable_cols] - train_mean) / train_std
        sel_vars[normalisable_cols] = (sel_vars[normalisable_cols] - train_mean) / train_std

    # Force all subjects to have the same number of samples (only for binary case)
    if cfg.eval.force_same_samples and not cfg.eval.is_multiclass:
        # Calculate minimum samples per activity per participant for each dataset
        # Training data
        train_min_samples_act1 = train_pp[train_labels == 1].value_counts().min()
        train_min_samples_act2 = train_pp[train_labels == 0].value_counts().min()
                
        # Selection data
        sel_min_samples_act1 = sel_pp[sel_labels == 1].value_counts().min()
        sel_min_samples_act2 = sel_pp[sel_labels == 0].value_counts().min()
        
        # Create balanced datasets by selecting samples per participant per activity
        balanced_train_indices = []
        for pp_id in train_pp.unique():
            # Get indices for each activity for this participant
            pp_act1_indices = train_vars.index[(train_pp == pp_id) & (train_labels == 1)][:train_min_samples_act1].tolist()
            pp_act2_indices = train_vars.index[(train_pp == pp_id) & (train_labels == 0)][:train_min_samples_act2].tolist()
            balanced_train_indices.extend(pp_act1_indices + pp_act2_indices)
        
        # Apply balancing to training data
        train_vars = train_vars.loc[balanced_train_indices]
        train_labels = train_labels.loc[balanced_train_indices]
        train_pp = train_pp.loc[balanced_train_indices]
        train_phone = train_phone.loc[balanced_train_indices]
        train_carryloc = train_carryloc.loc[balanced_train_indices]
                
        # Repeat for selection data
        balanced_sel_indices = []
        for pp_id in sel_pp.unique():
            pp_act1_indices = sel_vars.index[(sel_pp == pp_id) & (sel_labels == 1)][:sel_min_samples_act1].tolist()
            pp_act2_indices = sel_vars.index[(sel_pp == pp_id) & (sel_labels == 0)][:sel_min_samples_act2].tolist()
            balanced_sel_indices.extend(pp_act1_indices + pp_act2_indices)
        
        sel_vars = sel_vars.loc[balanced_sel_indices]
        sel_labels = sel_labels.loc[balanced_sel_indices]
        sel_pp = sel_pp.loc[balanced_sel_indices]
        sel_phone = sel_phone.loc[balanced_sel_indices]
        sel_carryloc = sel_carryloc.loc[balanced_sel_indices]


    # Final Checks
    if not cfg.eval.is_multiclass:
        n_clusters=None
        assert sum(train_labels==1) > 0, f"No samples of {activity_1} found in training data"
        assert sum(train_labels==0) > 0, f"No samples of {activity_2} found in training data"
        assert sum(sel_labels==1) > 0, f"No samples of {activity_1} found in selection data"
        assert sum(sel_labels==0) > 0, f"No samples of {activity_2} found in selection data"
        print(f"train data shape: {train_vars.shape} (1: {sum(train_labels==1)}, 2: {sum(train_labels==0)}); subjects present: {train_data['META_test_subject'].unique()}")
        print(f"sel data shape: {sel_vars.shape} (1: {sum(sel_labels==1)}, 2: {sum(sel_labels==0)}); subjects present: {sel_data['META_test_subject'].unique()}")

    else:
        le = LabelEncoder()
        le.fit(train_labels)
        train_labels = le.transform(train_labels)
        sel_labels = le.transform(sel_labels)
        le_name_mapping = {int(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        print("Label encoder mapping (original: encoded):", le_name_mapping)
        print(f"train data shape: {train_vars.shape}")
        print(f"sel data shape: {sel_vars.shape}")

    return train_vars, train_labels, train_pp, train_phone, train_carryloc, sel_vars, sel_labels, sel_pp, sel_phone, sel_carryloc, n_clusters

def setup_scorer(cfg, data, labels):
    if cfg.eval.is_multiclass:
        class_weights = None
        if cfg.scorer.use_class_weights:
            class_weights = get_class_weights(labels)
        if cfg.scorer.name == "CatBoost":
            model = CatBoost(cfg, is_multiclass=True, class_weights=class_weights)
        else:
            raise NotImplementedError(f"Scorer model {cfg.scorer.name} not implemented for Multiclass")

    else:
        if cfg.scorer.name == "XGBoost":
            model = XGBoost(cfg)
        elif cfg.scorer.name == "RandomForest":
            model = RandomForest(cfg)
        elif cfg.scorer.name == "CatBoost":
            model = CatBoost(cfg)
        elif cfg.scorer.name == "DecisionTree":
            model = DecisionTree(cfg)
        else:
            raise NotImplementedError(f"Scorer model {cfg.scorer.name} not implemented")
    
    # Preprocessor one-hot encodes categorical variables (if applicable)
    cat_preprocessor = setup_categorical_encoder(cfg, data)
    impute_preprocessor = setup_mean_imputer(cfg, data)

    scorer = Pipeline([
        ('imputer', impute_preprocessor),
        ('preprocessor', cat_preprocessor),
        ('model', model)
    ])

    return scorer

def setup_categorical_encoder(cfg, data):
    numerical_vars = [col for col in data.columns if col in cfg.data.numerical_vars]
    categorical_vars = [col for col in data.columns if col in cfg.data.categorical_vars]
    
    if len(categorical_vars) > 0:
        if cfg.scorer.one_hot:
            cat_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_vars),
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_vars)
                ])
        else:
            cat_preprocessor = 'passthrough'
    else: 
        cat_preprocessor = 'passthrough'
    
    return cat_preprocessor

def setup_mean_imputer(cfg, data):
    """
    Sets up a mean imputer for numerical variables in the dataset.
    If no numerical variables are specified, it returns a passthrough transformer.
    """
    if not hasattr(cfg.scorer, 'impute_type'):
        return 'passthrough'
    elif cfg.scorer.impute_type != 'mean':
        raise NotImplementedError(f"Imputer type {cfg.scorer.impute_type} not implemented")
    # Mean imputer (for numerical, modal for categorical)
    else:
        numerical_vars = [col for col in data.columns if col in cfg.data.numerical_vars]
        categorical_vars = [col for col in data.columns if col in cfg.data.categorical_vars]

        # Impute numerical variables with mean strategy
        num_imputer = SimpleImputer(strategy='mean')        
        # Impute categorical variables with most frequent value (mode)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        if len(numerical_vars) > 0 and len(categorical_vars) > 0:
            # Both numerical and categorical variables need imputation
            imputer = ColumnTransformer(
            transformers=[
                ('num_imp', num_imputer, numerical_vars),
                ('cat_imp', cat_imputer, categorical_vars)
            ])
        elif len(numerical_vars) > 0:
            # Only numerical variables need imputation
            imputer = num_imputer
        elif len(categorical_vars) > 0:
            # Only categorical variables need imputation
            imputer = cat_imputer
        else:
            imputer = 'passthrough'
    
        return imputer

def setup_calibrator(cfg):
    if cfg.eval.is_multiclass:
        class_indices = list(set(cfg.eval.expert_cluster_assignment.values()))

        if cfg.calibrator.name == "KDE":
            calibrator = MultiClassKDECalibrator(bandwidth=cfg.calibrator.bandwidth, class_indices=class_indices)
        elif cfg.calibrator.name == "LogReg":
            calibrator = MultiClassLogisticCalibrator(class_indices=class_indices, method=cfg.calibrator.method, solver=cfg.calibrator.solver)
        else:
            raise NotImplementedError(f"Calibrator {cfg.calibrator.name} not implemented for multiclass")
        
        if cfg.calibrator.bounded:
            calibrator = MultiClassELUBbounder(calibrator)

    else:
        if cfg.calibrator.name == "KDE":
            calibrator = lir.KDECalibrator(bandwidth=cfg.calibrator.bandwidth)
        elif cfg.calibrator.name == "LogReg":
            calibrator = lir.LogitCalibrator()
        elif cfg.calibrator.name == "Gaussian":
            calibrator = lir.GaussianCalibrator()
        else:
            raise NotImplementedError(f"Calibrator {cfg.calibrator.name} not implemented")

        if cfg.calibrator.bounded:
            calibrator = lir.ELUBbounder(calibrator)
    
    return calibrator

def compute_cmxe(log_likelihoods, classes):
    """
    Calculates the Multiclass-Cross-Entropy 
    :param log_likelihoods: log likelihoods for each class, shape n x K
    :param classes: list of classes, length n
    """

    assert log_likelihoods.shape[0] == len(classes), "log_likelihoods and classes must have the same length"
    assert log_likelihoods.ndim == 2, "log_likelihoods must be a 2D array"
    
    unique_classes = np.unique(classes)
    likelihood_classes = {classi: log_likelihoods[np.array(classes) == classi] for classi in unique_classes}        

    cmxe = 0.0
    for classi in unique_classes:
        lls = likelihood_classes[classi]
        class_total = 0
        for t in range(len(lls)):
            numerator = np.sum(np.exp(lls[t]))
            denominator = np.exp(lls[t,classi])
            class_total += np.log2(numerator / (denominator + 1e-8))
        cmxe += class_total / len(lls)

    cmxe /= len(unique_classes)
    return cmxe

def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = Counter(y)
    
    num_samples = len(y)
    weights = {}
    for idx in counter.keys():
        weights[idx] = 0.5 / (counter[idx] / num_samples)
    return weights


# ------------ Calibrated Scorer Custom Implementation ------------ #
# Copied CalibratedScorer from lir package, with some added functions for our implementation
class CalibratedScorer:
    """
    LR system which produces calibrated LRs using a scorer and a calibrator.

    The scorer is an object that transforms instance features into a scalar value (a "score"). The scorer may either
    be:
     - an estimator (e.g. `sklearn.linear_model.LogisticRegression`) object which implements `fit` and `predict_proba`;
     - a transformer object which implements `transform` and optionally `fit`; or
     - a distance function (callable) that takes paired instances as its arguments and returns a distance for each pair.

    The scorer can also be a composite object such as a `sklearn.pipeline.Pipeline`. If the scorer is an estimator, the
    probabilities it produces are transformed to their log odds.

    The calibrator is an object that transforms instance scores to LRs.

    This class supports `fit`, which fits the scorer and the calibrator on the same data. If only the calibrator is to
    be fit, use `fit_calibrator`. Both the scorer and the calibrator can be accessed by their attributes `scorer` and
    `calibrator`.
    """
    def __init__(self, cfg, scorer, calibrator):
        """
        Constructor.

        Parameters
        ----------
        scorer an object that transforms features into scores.
        calibrator an object that transforms scores into LRs.
        """
        self.cfg = cfg
        # create transformer changes probabilities to log odds (optionally) and calls .predict_proba on scorer when .fit is called
        if scorer is not None:
            self.scorer = _create_transformer(scorer)
        else:
            self.scorer = None
        self.calibrator = calibrator

        # Full pipeline
        self.pipeline = Pipeline([
            ("scorer", self.scorer),
            ("reshape", FunctionTransformer(self._reshape)), # makes reshape a transformer
            ("calibrator", self.calibrator)
        ])

    @staticmethod
    def _reshape(X):
        if len(X.shape) == 1:
            return X
        else:
            assert len(X) == X.shape[0], f"array has bad dimensions: all dimensions but the first should be 1; found {X.shape}"
            return X.reshape(-1)

    def fit(self, X, y):
        """
        Fits both the scorer and the calibrator on the same data.

        Parameters
        ----------
        X the feature vectors of the instances
        y the instance labels

        Returns
        -------
        `self`
        """
        self.pipeline.fit(X, y)
        return self

    def predict_class(self, X):
        return self.scorer.predict(X)
    
    def fit_calibrator(self, X, y):
        """
        Fits the calibrator without modifying the scorer.

        The arguments are the same as in `fit`. Before calibrating, this method transforms the feature vectors into
        scores by calling `transform` on the scorer.

        Parameters
        ----------
        X the feature vectors of the instances
        y the instance labels

        Returns
        -------
        `self`
        """
        X = self.scorer.transform(X)
        X = self._reshape(X)
        self.calibrator.fit(X, y)
        return self

    def predict_lr(self, X):
        """
        Compute LRs for instances.

        Parameters
        ----------
        X the feature vector of the instances

        Returns
        -------
        a vector of LRs
        """
        return self.pipeline.transform(X)

    
    def plot_tippett(self, lrs, labels):
        lrs_0 = np.log10(lrs[labels == 0])
        lrs_1 = np.log10(lrs[labels == 1])
      
        xplot0 = np.linspace(np.min(lrs_0), np.max(lrs_0), 100)
        xplot1 = np.linspace(np.min(lrs_1), np.max(lrs_1), 100)
        perc0 = (sum(i >= xplot0 for i in lrs_0) / len(lrs_0)) * 100
        perc1 = (sum(i >= xplot1 for i in lrs_1) / len(lrs_1)) * 100

        # Create figure and axis
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(xplot1, perc1, color='b', label=r'LRs given $\mathregular{H_1}$')
        ax.plot(xplot0, perc0, color='r', label=r'LRs given $\mathregular{H_2}$')
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_xlabel('log(LR)')
        ax.set_ylabel('Cumulative proportion')
        ax.legend()

        return ax

def _create_transformer(scorer):
    if hasattr(scorer, "transform"):
        return scorer
    elif hasattr(scorer, "predict_proba"):
        return EstimatorTransformer(scorer, transform_probabilities=to_log_odds)
    elif callable(scorer):
        return ComparisonFunctionTransformer(scorer)
    else:
        raise NotImplementedError("`scorer` argument must either be callable or implement at least one of `transform`, `predict_proba`")

class ComparisonFunctionTransformer(FunctionTransformer):
    """
    A wrapper for a distance function to make it behave like a transformer.

    This is essentially a `FunctionTransformer` except that it expects a function takes two arguments of the same
    dimensions: `X` and `Y`, where each row in `X` is compared to the row with the same index in `Y`.

    See: the `paired_*` distance functions in `sklearn.metrics.pairwise`.
    """
    def __init__(self, distance_function: Callable):
        self.distance_function = distance_function
        super().__init__(self.transformer_function)

    def transformer_function(self, X):
        if len(X.shape) == 2:
            return self.distance_function(X)
        elif len(X.shape) == 3:
            vectors = [X[:, :, i] for i in range(X.shape[2])]
            return self.distance_function(*vectors)
        else:
            raise ValueError(f"unexpected input shape; expected: (*, *) or (*, *, *); found: {X.shape}")

def to_log_odds(p):
    with np.errstate(divide='ignore'):
        complement = 1 - p
        return np.log10(p + 1e-8) - np.log10(complement + 1e-8)

class EstimatorTransformer(TransformerMixin):
    """
    A wrapper for an estimator to make it behave like a transformer.

    In particular, it implements `transform` by calling `predict_proba` on the underlying estimator, and transforming
    the probabilities to their corresponding log odds value. Optionally, an alternative transformation function can be
    specified.
    """
    def __init__(self, estimator, transform_probabilities: Optional[Callable] = to_log_odds):
        self.estimator = estimator
        self.transform_probabilities = transform_probabilities

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.transform_probabilities(self.estimator.predict_proba(X)[:, 1])

    def __getattr__(self, item):
        return getattr(self.estimator, item)
# ------------ /Calibrated Scorer Custom Implementation ------------ #

# ----------------- Multiclass Calibrators (adapted from lir package) ----------------- #
class MultiClassLogisticCalibrator(BaseEstimator, TransformerMixin):
    def __init__(self, class_indices: List, class_weights=None, method='multiclass', solver='lbfgs'):
        """
        method: 'multiclass' for multinomial logistic regression, 'classwise' for one-vs-rest (follows definitions of Silva Filho et al. 2023)
        """
        self.class_indices = class_indices
        if class_weights is not None:
            self.class_weights = {idx: 1/weight for idx, weight in zip(class_indices, class_weights)}
        else:
            self.class_weights = None
        self.K = len(class_indices)
        self.method = method
        self.solver = solver
        if self.method == "classwise":
            self.log_curves = [None] * self.K
        elif self.method == "multiclass":
            self.log_reg = LogisticRegression(solver=self.solver, class_weight=self.class_weights)
        else:
            raise NotImplementedError(f"Method {method} not implemented for MultiClassLogisticCalibrator")
        
    def fit(self, X, y):
        if self.method == "classwise":
            for class_idx in self.class_indices:
                binary_y = (y == class_idx).astype(int)  # Convert to binary labels for one-vs-rest
                self.log_curves[class_idx] = LogisticRegression(solver=self.solver).fit(X.reshape(-1,1), binary_y)
        elif self.method == "multiclass":
            self.log_reg.fit(X.reshape(-1, 1), y)
        return self

    def transform(self, X):
        """
        X: array of scorer outputs, shape n
        Outputs a n length array of log-likelihood ratios, one against the rest, where the one is chosen by the true class
        self.log_likelihoods will contain the log likelihoods for each class (n x K)
        """
        if self.method == "classwise":
            log_likelihoods = np.empty((len(X), self.K))
            for i in range(self.K):
                binary_scores = self.log_curves[i].predict_log_proba(X.reshape(-1,1))[:, 1]
                log_likelihoods[:, i] = binary_scores
            self.log_likelihoods = log_likelihoods
        elif self.method == "multiclass":
            log_likelihoods = self.log_reg.predict_log_proba(X.reshape(-1,1))
            self.log_likelihoods = log_likelihoods

        # Get the log-likelihoods of the true classes using advanced indexing
        # Find the maximum log-likelihood for each sample
        max_class_indices = np.argmax(log_likelihoods, axis=1)
        max_class_log_likelihoods = log_likelihoods[range(len(X)), max_class_indices]
        # Calculate the log-likelihood of all other classes combined (for one-vs-rest)
        other_classes_log_likelihoods = np.log(np.sum(np.exp(log_likelihoods), axis=1) - np.exp(max_class_log_likelihoods))
        # Calculate the log-likelihood ratio: max class vs all other classes
        likelihood_ratios = max_class_log_likelihoods - other_classes_log_likelihoods

        # Create and return array with both likelihood ratios and the corresponding max class indices
        return np.column_stack((likelihood_ratios, max_class_indices))


class LRbounder(ABC, BaseEstimator, TransformerMixin):
    """
    Class that, given an LR system, outputs the same LRs as the system but bounded by lower and upper bounds.
    """

    def __init__(self, first_step_calibrator, also_fit_calibrator=True):
        """
        a calibrator should be provided (optionally already fitted to data). This calibrator is called on scores,
        the resulting LRs are then bounded. If also_fit_calibrator, the first step calibrator will be fit on the same
        data used to derive the bounds
        :param first_step_calibrator: the calibrator to use. Should already have been fitted if also_fit_calibrator is False
        :param also_fit_calibrator: whether to also fit the first step calibrator when calling fit
        """

        self.first_step_calibrator = first_step_calibrator
        self.also_fit_calibrator = also_fit_calibrator
        self._lower_lr_bound = None
        self._upper_lr_bound = None
        if not also_fit_calibrator:
            # check the model was fitted.
            try:
                first_step_calibrator.transform(np.array([0.5]))
            except NotFittedError:
                print('calibrator should have been fit when setting also_fit_calibrator = False!')

    @abstractmethod
    def calculate_bounds(self, lrs, y):
        raise NotImplementedError

    def fit(self, X, y):
        """
        """
        if self.also_fit_calibrator:
            self.first_step_calibrator.fit(X, y)
        lrs = self.first_step_calibrator.transform(X)

        y = np.asarray(y).squeeze()
        self._lower_lr_bound, self._upper_lr_bound = self.calculate_bounds(lrs, y)
        return self

    def transform(self, X):
        """
        a transform entails calling the first step calibrator and applying the bounds found
        """
        unadjusted_lrs = np.array(self.first_step_calibrator.transform(X))
        self.log_likelihoods = self.first_step_calibrator.log_likelihoods
        lower_adjusted_lrs = np.where(self._lower_lr_bound < unadjusted_lrs, unadjusted_lrs, self._lower_lr_bound)
        adjusted_lrs = np.where(self._upper_lr_bound > lower_adjusted_lrs, lower_adjusted_lrs, self._upper_lr_bound)
        return adjusted_lrs

    @property
    def p0(self):
        return self.first_step_calibrator.p0

    @property
    def p1(self):
        return self.first_step_calibrator.p1


class MultiClassELUBbounder(LRbounder):
    """
    Class that, given an LR system, outputs the same LRs as the system but bounded by the Empirical Upper and Lower
    Bounds as described in
    P. Vergeer, A. van Es, A. de Jongh, I. Alberink, R.D. Stoel,
    Numerical likelihood ratios outputted by LR systems are often based on extrapolation:
    when to stop extrapolating?
    Sci. Justics 56 (2016) 482-491

    # MATLAB code from the authors:

    # clear all; close all;
    # llrs_hp=csvread('...');
    # llrs_hd=csvread('...');
    # start=-7; finish=7;
    # rho=start:0.01:finish; theta=10.^rho;
    # nbe=[];
    # for k=1:length(rho)
    #     if rho(k)<0
    #         llrs_hp=[llrs_hp;rho(k)];
    #         nbe=[nbe;(theta(k)^(-1))*mean(llrs_hp<=rho(k))+...
    #             mean(llrs_hd>rho(k))];
    #     else
    #         llrs_hd=[llrs_hd;rho(k)];
    #         nbe=[nbe;theta(k)*mean(llrs_hd>=rho(k))+...
    #             mean(llrs_hp<rho(k))];
    #     end
    # end
    # plot(rho,-log10(nbe)); hold on;
    # plot([start finish],[0 0]);
    # a=rho(-log10(nbe)>0);
    # empirical_bounds=[min(a) max(a)]
    """

    def calculate_bounds(self, lrs, y):

        lower_lr_bound, upper_lr_bound = elub(lrs, y, add_misleading=1)
        return lower_lr_bound, upper_lr_bound



def elub(lrs, y, add_misleading=1, step_size=.01, substitute_extremes=(np.exp(-20), np.exp(20))):
    """
    Returns the empirical upper and lower bound LRs (ELUB LRs).

    :param lrs: an array with LRs in first dim, index of multiclass prediction
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param add_misleading: the number of consequential misleading LRs to be added
        to both sides (labels 0 and 1)
    :param step_size: required accuracy on a natural logarithmic scale
    :param substitute_for_extremes (tuple of scalars): substitute for extreme LRs, i.e.
        LRs of 0 and inf are substituted by these values
    """

    # remove LRs of 0 and infinity
    sanitized_lrs = lrs[:,0]
    sanitized_lrs[sanitized_lrs < substitute_extremes[0]] = substitute_extremes[0]
    sanitized_lrs[sanitized_lrs > substitute_extremes[1]] = substitute_extremes[1]

    # determine the range of LRs to be considered
    llrs = np.log(sanitized_lrs)
    log_lr_threshold_range = (min(0, np.min(llrs)), max(0, np.max(llrs))+step_size)
    lr_threshold = np.exp(np.arange(*log_lr_threshold_range, step_size))

    adjusted_ys = (lrs[:,1] == y)
    eu_neutral = calculate_expected_utility(np.ones(len(sanitized_lrs)), adjusted_ys, lr_threshold)
    eu_system = calculate_expected_utility(sanitized_lrs, adjusted_ys, lr_threshold, add_misleading)
    eu_ratio = eu_neutral / eu_system

    # find threshold LRs which have utility ratio < 1 (only utility ratio >= 1 is acceptable)
    eu_negative_left = lr_threshold[(lr_threshold <= 1) & (eu_ratio < 1)]
    eu_negative_right = lr_threshold[(lr_threshold >= 1) & (eu_ratio < 1)]

    lower_bound = np.max(eu_negative_left * np.exp(step_size), initial=np.min(lr_threshold))
    upper_bound = np.min(eu_negative_right / np.exp(step_size), initial=np.max(lr_threshold))

    # Check for bounds on the wrong side of 1. This may occur for badly
    # performing LR systems, e.g. if expected utility is always below neutral.
    lower_bound = min(lower_bound, 1)
    upper_bound = max(upper_bound, 1)

    return lower_bound, upper_bound


def calculate_expected_utility(lrs, y, threshold_lrs, add_misleading=0):
    """
    Calculates the expected utility of a set of LRs for a given threshold.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param threshold_lrs: an array of threshold lrs: minimum LR for acceptance
    :returns: an array of utility values, one element for each threshold LR
    """
    m_accept = lrs.reshape(len(lrs), 1) > threshold_lrs.reshape(1, len(threshold_lrs))

    if add_misleading > 0:
        n_elems = len(threshold_lrs) * add_misleading
        m_accept = np.concatenate([m_accept,
                   np.zeros(n_elems).reshape(add_misleading, len(threshold_lrs)),
                   np.ones(n_elems).reshape(add_misleading, len(threshold_lrs))])
        y = np.concatenate([y, np.ones(add_misleading), np.zeros(add_misleading)])

    eu = 1 - np.average(m_accept[y==1], axis=0) + threshold_lrs * np.average(m_accept[y==0], axis=0)
    return eu

class MultiClassKDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, allowing for K > 2 classes (score distributions).
    Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, class_indices: List, bandwidth: Union[Callable, str, float, Tuple[float, float]] = None):
        """
        :param bandwidth:
            * If bandwidth has a float value, this value is used as the bandwidth for both distributions.
            * If bandwidth is a tuple, it should contain two floating point values: the bandwidth for the distribution
              of the classes with labels 0 and 1, respectively.
            * If bandwidth has the str value "silverman", Silverman's rule of thumb is used as the bandwidth for both
              distributions separately.
            * If bandwidth is callable, it should accept two arguments, `X` and `y`, and return a tuple of two values
              which are the bandwidths for the two distributions.
        """
        if bandwidth is None:
            warnings.warn("missing bandwidth argument for KDE, defaulting to silverman (default argument will be removed in the future)")
            bandwidth = "silverman"
        self.bandwidth: Callable = self._parse_bandwidth(bandwidth)
        self.class_indices = class_indices
        self.K = len(class_indices) 
        self._kdes: List[Optional[KernelDensity]] = [None] * self.K  # List of KDE for each class
        self.numerator, self.denominator = None, None

    @staticmethod
    def bandwidth_silverman(X, y):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        bandwidth = []
        for label in np.unique(y):
            values = X[y == label]
            std = np.std(values)
            if std == 0:
                # can happen eg if std(values) = 0
                warnings.warn('silverman bandwidth cannot be calculated if standard deviation is 0', RuntimeWarning)
                LOG.info('found a silverman bandwidth of 0 (using dummy value)')
                std = 1

            v = math.pow(std, 5) / len(values) * 4. / 3
            bandwidth.append(math.pow(v, .2))

        return bandwidth

    def fit(self, X, y):
        # check if data is sane
        check_misleading_Inf_negInf(X, y)

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in transform function
        Xis = Xy_to_Xn(X, y, classes=self.class_indices)
        for Xi in Xis:
            Xi = Xi.reshape(-1, 1)

        bandwidths = self.bandwidth(X, y)
        for i in range(self.K):
            self._kdes[i] = KernelDensity(kernel='gaussian', bandwidth=bandwidths[i]).fit(Xis[i].reshape(-1, 1))
        return self

    def transform(self, X):
        """
        X: array of scorer outputs, shape n x K, score for each class of every sample
        Using the fitted KDEs, outputs the likelihood for each class
        # Outputs a n x K array of log likelihoods
        Outputs a n length array of log-likelihood ratios, one against the rest, where the one is chosen by the true class
        """
     
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))

        # get inf and neginf
        assert np.sum(np.isinf(X)) == 0, "X should not contain inf or neginf values"
        assert np.sum(np.isnan(X)) == 0, "X should not contain NaN"

        # perform KDE as usual
        X = X.reshape(-1, 1)
        log_likelihoods = np.empty((len(X), self.K))
        for i in range(self.K):
            log_likelihoods[:, i] = self._kdes[i].score_samples(X)

        self.log_likelihoods = log_likelihoods

        # Get the log-likelihoods of the true classes using advanced indexing
        # Find the maximum log-likelihood for each sample
        max_class_indices = np.argmax(log_likelihoods, axis=1)
        max_class_log_likelihoods = log_likelihoods[range(len(X)), max_class_indices]
        # Calculate the log-likelihood of all other classes combined (for one-vs-rest)
        other_classes_log_likelihoods = np.log(np.sum(np.exp(log_likelihoods), axis=1) - np.exp(max_class_log_likelihoods))
        # Calculate the log-likelihood ratio: max class vs all other classes
        likelihood_ratios = max_class_log_likelihoods - other_classes_log_likelihoods

        return likelihood_ratios

    @staticmethod
    def _parse_bandwidth(bandwidth: Union[Callable, float, Tuple[float, float]]) \
            -> Callable:
        """
        Returns bandwidth as a tuple of two (optional) floats.
        Extrapolates a single bandwidth
        :param bandwidth: provided bandwidth
        :return: bandwidth used for kde0, bandwidth used for kde1
        """
        assert bandwidth is not None, "KDE requires a bandwidth argument"
        if callable(bandwidth):
            return bandwidth
        elif bandwidth == "silverman":
            return MultiClassKDECalibrator.bandwidth_silverman
        elif isinstance(bandwidth, str):
            raise ValueError(f"invalid input for bandwidth: {bandwidth}")
        elif isinstance(bandwidth, Sized):
            assert len(bandwidth) == 2, f"bandwidth should have two elements; found {len(bandwidth)}; bandwidth = {bandwidth}"
            return lambda X, y: bandwidth
        else:
            return lambda X, y: (0+bandwidth, bandwidth)

# ----------------- /Multiclass Calibrators ----------------- #
