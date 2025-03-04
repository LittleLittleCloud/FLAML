# !
#  * Copyright (c) FLAML authors. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
import time
import numpy as np
import pandas as pd
from typing import Union, Callable, TypeVar, Optional, Tuple

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    log_loss,
    average_precision_score,
    f1_score,
    mean_absolute_percentage_error,
    ndcg_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    StratifiedGroupKFold,
)
from flaml.automl.model import (
    XGBoostSklearnEstimator,
    XGBoost_TS,
    XGBoostLimitDepthEstimator,
    XGBoostLimitDepth_TS,
    RandomForestEstimator,
    RF_TS,
    LGBMEstimator,
    LGBM_TS,
    LRL1Classifier,
    LRL2Classifier,
    CatBoostEstimator,
    ExtraTreesEstimator,
    ExtraTrees_TS,
    KNeighborsEstimator,
    Prophet,
    ARIMA,
    SARIMAX,
    TransformersEstimator,
    TemporalFusionTransformerEstimator,
    TransformersEstimatorModelSelection,
)
from flaml.automl.data import CLASSIFICATION, group_counts, TS_FORECAST
from flaml.automl.model import BaseEstimator
import logging

logger = logging.getLogger(__name__)
EstimatorSubclass = TypeVar("EstimatorSubclass", bound=BaseEstimator)

sklearn_metric_name_set = {
    "r2",
    "rmse",
    "mae",
    "mse",
    "accuracy",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_weighted",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
    "log_loss",
    "mape",
    "f1",
    "ap",
    "ndcg",
    "micro_f1",
    "macro_f1",
}
huggingface_metric_to_mode = {
    "accuracy": "max",
    "bertscore": "max",
    "bleu": "max",
    "bleurt": "max",
    "cer": "min",
    "chrf": "min",
    "code_eval": "max",
    "comet": "max",
    "competition_math": "max",
    "coval": "max",
    "cuad": "max",
    "f1": "max",
    "gleu": "max",
    "google_bleu": "max",
    "matthews_correlation": "max",
    "meteor": "max",
    "pearsonr": "max",
    "precision": "max",
    "recall": "max",
    "rouge": "max",
    "sacrebleu": "max",
    "sari": "max",
    "seqeval": "max",
    "spearmanr": "max",
    "ter": "min",
    "wer": "min",
}
huggingface_submetric_to_metric = {"rouge1": "rouge", "rouge2": "rouge"}


def get_estimator_class(task: str, estimator_name: str) -> EstimatorSubclass:
    """Given a task and an estimator name, return the relevant flaml-wrapped estimator class

    NOTE: See why the return type is declarad by using TypeVar here on the mypy doc
    https://mypy.readthedocs.io/en/stable/kinds_of_types.html#the-type-of-class-objects
    """
    # when adding a new learner, need to add an elif branch
    if "xgboost" == estimator_name:
        estimator_class = XGBoost_TS if task in TS_FORECAST else XGBoostSklearnEstimator
    elif "xgb_limitdepth" == estimator_name:
        estimator_class = (
            XGBoostLimitDepth_TS if task in TS_FORECAST else XGBoostLimitDepthEstimator
        )
    elif "rf" == estimator_name:
        estimator_class = RF_TS if task in TS_FORECAST else RandomForestEstimator
    elif "lgbm" == estimator_name:
        estimator_class = LGBM_TS if task in TS_FORECAST else LGBMEstimator
    elif "lrl1" == estimator_name:
        estimator_class = LRL1Classifier
    elif "lrl2" == estimator_name:
        estimator_class = LRL2Classifier
    elif "catboost" == estimator_name:
        estimator_class = CatBoostEstimator
    elif "extra_tree" == estimator_name:
        estimator_class = ExtraTrees_TS if task in TS_FORECAST else ExtraTreesEstimator
    elif "kneighbor" == estimator_name:
        estimator_class = KNeighborsEstimator
    elif "prophet" in estimator_name:
        estimator_class = Prophet
    elif estimator_name == "arima":
        estimator_class = ARIMA
    elif estimator_name == "sarimax":
        estimator_class = SARIMAX
    elif estimator_name == "transformer":
        estimator_class = TransformersEstimator
    elif estimator_name == "tft":
        estimator_class = TemporalFusionTransformerEstimator
    elif estimator_name == "transformer_ms":
        estimator_class = TransformersEstimatorModelSelection
    else:
        raise ValueError(
            estimator_name + " is not a built-in learner. "
            "Please use AutoML.add_learner() to add a customized learner."
        )
    return estimator_class


def metric_loss_score(
    metric_name: str,
    y_processed_predict,
    y_processed_true,
    labels=None,
    sample_weight=None,
    groups=None,
):
    # y_processed_predict and y_processed_true are processed id labels if the original were the token labels
    if is_in_sklearn_metric_name_set(metric_name):
        return sklearn_metric_loss_score(
            metric_name,
            y_processed_predict,
            y_processed_true,
            labels,
            sample_weight,
            groups,
        )
    else:
        try:
            import datasets

            datasets_metric_name = huggingface_submetric_to_metric.get(
                metric_name, metric_name.split(":")[0]
            )
            metric = datasets.load_metric(datasets_metric_name)
            metric_mode = huggingface_metric_to_mode[datasets_metric_name]

            if metric_name.startswith("seqeval"):
                y_processed_true = [
                    [labels[tr] for tr in each_list] for each_list in y_processed_true
                ]
            elif metric in ("pearsonr", "spearmanr"):
                y_processed_true = (
                    y_processed_true.to_list()
                    if isinstance(y_processed_true, pd.Series)
                    else list(y_processed_true)
                )
            score_dict = metric.compute(
                predictions=y_processed_predict, references=y_processed_true
            )
            if "rouge" in metric_name:
                score = score_dict[metric_name].mid.fmeasure
            elif metric_name.startswith("seqeval"):
                metric_submetric_names = metric_name.split(":")
                score = score_dict[
                    metric_submetric_names[1]
                    if len(metric_submetric_names) > 1
                    else "overall_accuracy"
                ]
            else:
                score = score_dict[metric_name]
        except ImportError:
            raise ValueError(
                metric_name
                + " is not an built-in sklearn metric and nlp is not installed. "
                "Currently built-in sklearn metrics are: "
                "r2, rmse, mae, mse, accuracy, roc_auc, roc_auc_ovr, roc_auc_ovo,"
                "log_loss, mape, f1, micro_f1, macro_f1, ap. "
                "If the metric is an nlp metric, please pip install flaml[nlp] ",
                "or pass a customized metric function to AutoML.fit(metric=func)",
            )
        # If the metric is not found from huggingface dataset metric list (i.e., FileNotFoundError)
        # ask the user to provide a custom metric
        except FileNotFoundError:
            raise ValueError(
                metric_name + " is neither an sklearn metric nor a huggingface metric. "
                "Currently built-in sklearn metrics are: "
                "r2, rmse, mae, mse, accuracy, roc_auc, roc_auc_ovr, roc_auc_ovo,"
                "log_loss, mape, f1, micro_f1, macro_f1, ap. "
                "Currently built-in huggingface metrics are: "
                + ", ".join(huggingface_metric_to_mode.keys())
                + ". Please pass a customized metric function to AutoML.fit(metric=func)"
            )
        if metric_mode == "max":
            return 1 - score
        else:
            return score


def is_in_sklearn_metric_name_set(metric_name: str):
    return metric_name.startswith("ndcg") or metric_name in sklearn_metric_name_set


def is_min_metric(metric_name: str):
    return (
        metric_name in ["rmse", "mae", "mse", "log_loss", "mape"]
        or huggingface_metric_to_mode.get(metric_name, None) == "min"
    )


def sklearn_metric_loss_score(
    metric_name: str,
    y_predict,
    y_true,
    labels=None,
    sample_weight=None,
    groups=None,
):
    """Loss using the specified metric.

    Args:
        metric_name: A string of the metric name, one of
            'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
            'roc_auc_ovo', 'roc_auc_weighted', 'roc_auc_ovo_weighted', 'roc_auc_ovr_weighted',
            'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'.
        y_predict: A 1d or 2d numpy array of the predictions which can be
            used to calculate the metric. E.g., 2d for log_loss and 1d
            for others.
        y_true: A 1d numpy array of the true labels.
        labels: A list or an array of the unique labels.
        sample_weight: A 1d numpy array of the sample weight.
        groups: A 1d numpy array of the group labels.

    Returns:
        score: A float number of the loss, the lower the better.
    """
    metric_name = metric_name.lower()

    if "r2" == metric_name:
        score = 1.0 - r2_score(y_true, y_predict, sample_weight=sample_weight)
    elif metric_name == "rmse":
        score = np.sqrt(
            mean_squared_error(y_true, y_predict, sample_weight=sample_weight)
        )
    elif metric_name == "mae":
        score = mean_absolute_error(y_true, y_predict, sample_weight=sample_weight)
    elif metric_name == "mse":
        score = mean_squared_error(y_true, y_predict, sample_weight=sample_weight)
    elif metric_name == "accuracy":
        score = 1.0 - accuracy_score(y_true, y_predict, sample_weight=sample_weight)
    elif metric_name == "roc_auc":
        score = 1.0 - roc_auc_score(y_true, y_predict, sample_weight=sample_weight)
    elif metric_name == "roc_auc_ovr":
        score = 1.0 - roc_auc_score(
            y_true, y_predict, sample_weight=sample_weight, multi_class="ovr"
        )
    elif metric_name == "roc_auc_ovo":
        score = 1.0 - roc_auc_score(
            y_true, y_predict, sample_weight=sample_weight, multi_class="ovo"
        )
    elif metric_name == "roc_auc_weighted":
        score = 1.0 - roc_auc_score(
            y_true, y_predict, sample_weight=sample_weight, average="weighted"
        )
    elif metric_name == "roc_auc_ovo_weighted":
        score = 1.0 - roc_auc_score(
            y_true,
            y_predict,
            sample_weight=sample_weight,
            average="weighted",
            multi_class="ovo",
        )
    elif metric_name == "roc_auc_ovr_weighted":
        score = 1.0 - roc_auc_score(
            y_true,
            y_predict,
            sample_weight=sample_weight,
            average="weighted",
            multi_class="ovr",
        )
    elif "log_loss" == metric_name:
        score = log_loss(y_true, y_predict, labels=labels, sample_weight=sample_weight)
    elif "mape" == metric_name:
        try:
            score = mean_absolute_percentage_error(y_true, y_predict)
        except ValueError:
            return np.inf
    elif "micro_f1" == metric_name:
        score = 1 - f1_score(
            y_true, y_predict, sample_weight=sample_weight, average="micro"
        )
    elif "macro_f1" == metric_name:
        score = 1 - f1_score(
            y_true, y_predict, sample_weight=sample_weight, average="macro"
        )
    elif "f1" == metric_name:
        score = 1 - f1_score(y_true, y_predict, sample_weight=sample_weight)
    elif "ap" == metric_name:
        score = 1 - average_precision_score(
            y_true, y_predict, sample_weight=sample_weight
        )
    elif "ndcg" in metric_name:
        if "@" in metric_name:
            k = int(metric_name.split("@", 1)[-1])
            counts = group_counts(groups)
            score = 0
            psum = 0
            for c in counts:
                score -= ndcg_score(
                    np.asarray([y_true[psum : psum + c]]),
                    np.asarray([y_predict[psum : psum + c]]),
                    k=k,
                )
                psum += c
            score /= len(counts)
            score += 1
        else:
            score = 1 - ndcg_score([y_true], [y_predict])
    return score


def get_y_pred(estimator, X, eval_metric, obj):
    if eval_metric in ["roc_auc", "ap", "roc_auc_weighted"] and "binary" in obj:
        y_pred_classes = estimator.predict_proba(X)
        y_pred = y_pred_classes[:, 1] if y_pred_classes.ndim > 1 else y_pred_classes
    elif eval_metric in [
        "log_loss",
        "roc_auc",
        "roc_auc_ovr",
        "roc_auc_ovo",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr_weighted",
    ]:
        y_pred = estimator.predict_proba(X)
    else:
        y_pred = estimator.predict(X)
    return y_pred


def _eval_estimator(
    config,
    estimator,
    X_train,
    y_train,
    X_val,
    y_val,
    weight_val,
    groups_val,
    eval_metric: Union[str, Callable],
    obj,
    labels=None,
    log_training_metric=False,
    fit_kwargs: Optional[dict] = None,
):
    if fit_kwargs is None:
        fit_kwargs = {}
    if isinstance(eval_metric, str):
        pred_start = time.time()
        val_pred_y = get_y_pred(estimator, X_val, eval_metric, obj)
        pred_time = (time.time() - pred_start) / X_val.shape[0]

        val_loss = metric_loss_score(
            eval_metric,
            y_processed_predict=val_pred_y,
            y_processed_true=y_val,
            labels=labels,
            sample_weight=weight_val,
            groups=groups_val,
        )
        metric_for_logging = {"pred_time": pred_time}
        if log_training_metric:
            train_pred_y = get_y_pred(estimator, X_train, eval_metric, obj)
            metric_for_logging["train_loss"] = metric_loss_score(
                eval_metric,
                train_pred_y,
                y_train,
                labels,
                fit_kwargs.get("sample_weight"),
                fit_kwargs.get("groups"),
            )
    else:  # customized metric function
        val_loss, metric_for_logging = eval_metric(
            X_val,
            y_val,
            estimator,
            labels,
            X_train,
            y_train,
            weight_val,
            fit_kwargs.get("sample_weight"),
            config,
            groups_val,
            fit_kwargs.get("groups"),
        )
        pred_time = metric_for_logging.get("pred_time", 0)
        val_pred_y = None
        # eval_metric may return val_pred_y but not necessarily. Setting None for now.
    return val_loss, metric_for_logging, pred_time, val_pred_y


def get_val_loss(
    config,
    estimator: EstimatorSubclass,
    X_train,
    y_train,
    X_val,
    y_val,
    weight_val,
    groups_val,
    eval_metric: Union[str, Callable],
    obj,
    labels=None,
    budget=None,
    log_training_metric=False,
    fit_kwargs: Optional[dict] = None,
    free_mem_ratio=0,
):
    if fit_kwargs is None:
        fit_kwargs = {}
    start = time.time()
    # if groups_val is not None:
    #     fit_kwargs['groups_val'] = groups_val
    #     fit_kwargs['X_val'] = X_val
    #     fit_kwargs['y_val'] = y_val
    estimator.fit(X_train, y_train, budget, free_mem_ratio, **fit_kwargs)
    val_loss, metric_for_logging, pred_time, _ = _eval_estimator(
        config,
        estimator,
        X_train,
        y_train,
        X_val,
        y_val,
        weight_val,
        groups_val,
        eval_metric,
        obj,
        labels,
        log_training_metric,
        fit_kwargs,
    )
    if hasattr(estimator, "intermediate_results"):
        metric_for_logging["intermediate_results"] = estimator.intermediate_results
    train_time = time.time() - start
    return val_loss, metric_for_logging, train_time, pred_time


def default_cv_score_agg_func(val_loss_folds, log_metrics_folds):
    metric_to_minimize = sum(val_loss_folds) / len(val_loss_folds)
    metrics_to_log = None
    for single_fold in log_metrics_folds:
        if metrics_to_log is None:
            metrics_to_log = single_fold
        elif isinstance(metrics_to_log, dict):
            metrics_to_log = {k: metrics_to_log[k] + v for k, v in single_fold.items()}
        else:
            metrics_to_log += single_fold
    if metrics_to_log:
        n = len(val_loss_folds)
        metrics_to_log = (
            {k: v / n for k, v in metrics_to_log.items()}
            if isinstance(metrics_to_log, dict)
            else metrics_to_log / n
        )
    return metric_to_minimize, metrics_to_log


def evaluate_model_CV(
    config: dict,
    estimator: EstimatorSubclass,
    X_train_all,
    y_train_all,
    budget,
    kf,
    task: str,
    eval_metric,
    best_val_loss,
    cv_score_agg_func=None,
    log_training_metric=False,
    fit_kwargs: Optional[dict] = None,
    free_mem_ratio=0,
):
    if fit_kwargs is None:
        fit_kwargs = {}
    if cv_score_agg_func is None:
        cv_score_agg_func = default_cv_score_agg_func
    start_time = time.time()
    val_loss_folds = []
    log_metric_folds = []
    metric = None
    train_time = pred_time = 0
    total_fold_num = 0
    n = kf.get_n_splits()
    X_train_split, y_train_split = X_train_all, y_train_all
    if task in CLASSIFICATION:
        labels = np.unique(y_train_all)
    else:
        labels = fit_kwargs.get(
            "label_list"
        )  # pass the label list on to compute the evaluation metric
    groups = None
    shuffle = getattr(kf, "shuffle", task not in TS_FORECAST)
    if isinstance(kf, RepeatedStratifiedKFold):
        kf = kf.split(X_train_split, y_train_split)
    elif isinstance(kf, (GroupKFold, StratifiedGroupKFold)):
        groups = kf.groups
        kf = kf.split(X_train_split, y_train_split, groups)
        shuffle = False
    elif isinstance(kf, TimeSeriesSplit):
        kf = kf.split(X_train_split, y_train_split)
    else:
        kf = kf.split(X_train_split)
    rng = np.random.RandomState(2020)
    budget_per_train = budget and budget / n
    if "sample_weight" in fit_kwargs:
        weight = fit_kwargs["sample_weight"]
        weight_val = None
    else:
        weight = weight_val = None
    for train_index, val_index in kf:
        if shuffle:
            train_index = rng.permutation(train_index)
        if isinstance(X_train_all, pd.DataFrame):
            X_train = X_train_split.iloc[train_index]
            X_val = X_train_split.iloc[val_index]
        else:
            X_train, X_val = X_train_split[train_index], X_train_split[val_index]
        y_train, y_val = y_train_split[train_index], y_train_split[val_index]
        estimator.cleanup()
        if weight is not None:
            fit_kwargs["sample_weight"], weight_val = (
                weight[train_index],
                weight[val_index],
            )
        if groups is not None:
            fit_kwargs["groups"] = (
                groups[train_index]
                if isinstance(groups, np.ndarray)
                else groups.iloc[train_index]
            )
            groups_val = (
                groups[val_index]
                if isinstance(groups, np.ndarray)
                else groups.iloc[val_index]
            )
        else:
            groups_val = None
        val_loss_i, metric_i, train_time_i, pred_time_i = get_val_loss(
            config,
            estimator,
            X_train,
            y_train,
            X_val,
            y_val,
            weight_val,
            groups_val,
            eval_metric,
            task,
            labels,
            budget_per_train,
            log_training_metric=log_training_metric,
            fit_kwargs=fit_kwargs,
            free_mem_ratio=free_mem_ratio,
        )
        if isinstance(metric_i, dict) and "intermediate_results" in metric_i.keys():
            del metric_i["intermediate_results"]
        if weight is not None:
            fit_kwargs["sample_weight"] = weight
        total_fold_num += 1
        val_loss_folds.append(val_loss_i)
        log_metric_folds.append(metric_i)
        train_time += train_time_i
        pred_time += pred_time_i
        if budget and time.time() - start_time >= budget:
            break
    val_loss, metric = cv_score_agg_func(val_loss_folds, log_metric_folds)
    n = total_fold_num
    pred_time /= n
    return val_loss, metric, train_time, pred_time


def compute_estimator(
    X_train,
    y_train,
    X_val,
    y_val,
    weight_val,
    groups_val,
    budget,
    kf,
    config_dic: dict,
    task: str,
    estimator_name: str,
    eval_method: str,
    eval_metric: Union[str, Callable],
    best_val_loss=np.Inf,
    n_jobs: Optional[
        int
    ] = 1,  # some estimators of EstimatorSubclass don't accept n_jobs. Should be None in that case.
    estimator_class: Optional[EstimatorSubclass] = None,
    cv_score_agg_func: Optional[callable] = None,
    log_training_metric: Optional[bool] = False,
    fit_kwargs: Optional[dict] = None,
    free_mem_ratio=0,
):
    if not fit_kwargs:
        fit_kwargs = {}

    estimator_class = estimator_class or get_estimator_class(task, estimator_name)
    estimator = estimator_class(
        **config_dic,
        task=task,
        n_jobs=n_jobs,
    )

    if isinstance(estimator, TransformersEstimator):
        # TODO: move the partial function to nlp
        fit_kwargs["metric"] = eval_metric
        fit_kwargs["X_val"] = X_val
        fit_kwargs["y_val"] = y_val

    if "holdout" == eval_method:
        val_loss, metric_for_logging, train_time, pred_time = get_val_loss(
            config_dic,
            estimator,
            X_train,
            y_train,
            X_val,
            y_val,
            weight_val,
            groups_val,
            eval_metric,
            task,
            labels=fit_kwargs.get(
                "label_list"
            ),  # pass the label list on to compute the evaluation metric
            budget=budget,
            log_training_metric=log_training_metric,
            fit_kwargs=fit_kwargs,
            free_mem_ratio=0,
        )
    else:
        val_loss, metric_for_logging, train_time, pred_time = evaluate_model_CV(
            config_dic,
            estimator,
            X_train,
            y_train,
            budget,
            kf,
            task,
            eval_metric,
            best_val_loss,
            cv_score_agg_func,
            log_training_metric=log_training_metric,
            fit_kwargs=fit_kwargs,
            free_mem_ratio=0,
        )

    if isinstance(estimator, TransformersEstimator):
        del fit_kwargs["metric"], fit_kwargs["X_val"], fit_kwargs["y_val"]

    return estimator, val_loss, metric_for_logging, train_time, pred_time


def train_estimator(
    config_dic: dict,
    X_train,
    y_train,
    task: str,
    estimator_name: str,
    n_jobs: Optional[
        int
    ] = 1,  # some estimators of EstimatorSubclass don't accept n_jobs. Should be None in that case.
    estimator_class: Optional[EstimatorSubclass] = None,
    budget=None,
    fit_kwargs: Optional[dict] = None,
    eval_metric=None,
    free_mem_ratio=0,
) -> Tuple[EstimatorSubclass, float]:
    start_time = time.time()
    estimator_class = estimator_class or get_estimator_class(task, estimator_name)
    estimator = estimator_class(
        **config_dic,
        task=task,
        n_jobs=n_jobs,
    )
    if not fit_kwargs:
        fit_kwargs = {}

    if isinstance(estimator, TransformersEstimator):
        fit_kwargs["metric"] = eval_metric

    if X_train is not None:
        train_time = estimator.fit(
            X_train, y_train, budget, free_mem_ratio, **fit_kwargs
        )
    else:
        estimator = estimator.estimator_class(**estimator.params)
    train_time = time.time() - start_time
    return estimator, train_time


def get_classification_objective(num_labels: int) -> str:
    if num_labels == 2:
        objective_name = "binary"
    else:
        objective_name = "multiclass"
    return objective_name


def norm_confusion_matrix(
    y_true: Union[np.array, pd.Series], y_pred: Union[np.array, pd.Series]
):
    """normalized confusion matrix.

    Args:
        estimator: A multi-class classification estimator.
        y_true: A numpy array or a pandas series of true labels.
        y_pred: A numpy array or a pandas series of predicted labels.

    Returns:
        A normalized confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    conf_mat = confusion_matrix(y_true, y_pred)
    norm_conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    return norm_conf_mat


def multi_class_curves(
    y_true: Union[np.array, pd.Series],
    y_pred_proba: Union[np.array, pd.Series],
    curve_func: Callable,
):
    """Binarize the data for multi-class tasks and produce ROC or precision-recall curves.

    Args:
        y_true: A numpy array or a pandas series of true labels.
        y_pred_proba: A numpy array or a pandas dataframe of predicted probabilites.
        curve_func: A function to produce a curve (e.g., roc_curve or precision_recall_curve).

    Returns:
        A tuple of two dictionaries with the same set of keys (class indices).
        The first dictionary curve_x stores the x coordinates of each curve, e.g.,
            curve_x[0] is an 1D array of the x coordinates of class 0.
        The second dictionary curve_y stores the y coordinates of each curve, e.g.,
            curve_y[0] is an 1D array of the y coordinates of class 0.
    """
    from sklearn.preprocessing import label_binarize

    classes = np.unique(y_true)
    y_true_binary = label_binarize(y_true, classes=classes)

    curve_x, curve_y = {}, {}
    for i in range(len(classes)):
        curve_x[i], curve_y[i], _ = curve_func(y_true_binary[:, i], y_pred_proba[:, i])
    return curve_x, curve_y
