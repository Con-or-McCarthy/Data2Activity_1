import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import catboost as cb
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils import load_data, split_train_select, setup_scorer, setup_calibrator, CalibratedScorer


def _get_feature_names_after_imputer(imputer, fallback_cols):
    """Return column names in the order they come out of the imputer step."""
    if imputer == "passthrough":
        return list(fallback_cols)
    if hasattr(imputer, "get_feature_names_out"):
        try:
            raw = imputer.get_feature_names_out()
            return [n.split("__", 1)[-1] for n in raw]
        except Exception:
            pass
    return list(fallback_cols)


def get_shap_values(cfg, lr_system, sel_vars):
    """Compute SHAP values via CatBoost's native ShapValues implementation."""
    scorer_pipeline = lr_system.scorer.estimator  # sklearn Pipeline
    imputer = scorer_pipeline.named_steps["imputer"]
    catboost_model = scorer_pipeline.named_steps["model"]

    # Apply imputation (preprocessor is passthrough for CatBoost)
    if imputer == "passthrough":
        X_imputed = sel_vars.values
    else:
        X_imputed = imputer.transform(sel_vars)

    feature_names = _get_feature_names_after_imputer(imputer, sel_vars.columns)
    X_df = pd.DataFrame(X_imputed, columns=feature_names)

    # Replicate CatBoost.fit Pool construction
    cat_features = [
        f for f in cfg.data.categorical_vars
        if f in cfg.data.vars_to_use and f in X_df.columns
    ]
    for col in cat_features:
        X_df[col] = X_df[col].astype(str)

    pool = cb.Pool(data=X_df, cat_features=cat_features)
    # Binary:     (n, n_features + 1)      – last col is bias
    # Multiclass: (n, n_classes, n_features + 1)
    shap_matrix = catboost_model.get_feature_importance(pool, type="ShapValues")

    if shap_matrix.ndim == 2:
        return shap_matrix[:, :-1], feature_names
    else:
        return shap_matrix[:, :, :-1], feature_names


def plot_shap(shap_values, feature_names, save_path="figs/shap_values.png"):
    if shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    top_n = min(20, len(feature_names))
    order = np.argsort(mean_abs)[-top_n:]  # ascending for barh

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    ax.barh(np.array(feature_names)[order], mean_abs[order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP plot to {save_path}")


def plot_confusion_matrix(labels, predictions, display_labels, save_path="figs/confusion_matrix.png"):
    cm = confusion_matrix(labels, predictions, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    size = max(4, len(display_labels) * 1.5)
    fig, ax = plt.subplots(figsize=(size, size))
    disp.plot(ax=ax, colorbar=False, xticks_rotation="vertical")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def train_select_analyse(cfg):
    paths = {"NFI": cfg.eval.nfi_data_path, "AUAS": cfg.eval.auas_data_path}
    train_path = paths[cfg.eval.train_data]
    test_dataset = "AUAS" if cfg.eval.train_data == "NFI" else "NFI"

    test_data = None
    if cfg.eval.setup == "cross":
        test_data = pd.read_csv(paths[test_dataset])
        print(f"Cross-dataset setup: training on {cfg.eval.train_data}, testing on {test_dataset}")

    if test_data is not None:
        sel_sets = [None]
    elif cfg.eval.do_cv:
        pp_list = cfg.eval.pp_list
        n = cfg.eval.sel_set_size
        sel_sets = [pp_list[i:i + n] for i in range(0, len(pp_list), n)]
    else:
        sel_sets = [cfg.eval.sel_pp]

    all_sel_vars = []
    all_labels = []
    all_preds = []
    last_lr_system = None

    print(f"Training data: {train_path}")
    print(f"Activities: {cfg.eval.activity_pair}\n")

    for sel_set in sel_sets:
        data = load_data(train_path)
        try:
            train_vars, train_labels, train_pp, train_phone, train_carryloc, \
            sel_vars, sel_labels, sel_pp, sel_phone, sel_carryloc, n_clusters = \
                split_train_select(cfg, data, sel_set, test_data=test_data)
        except Exception as e:
            print(f"Error for selection set {sel_set}: {e}. Skipping.")
            continue

        scorer = setup_scorer(cfg, train_vars, train_labels)
        calibrator = setup_calibrator(cfg)
        lr_system = CalibratedScorer(cfg, scorer, calibrator)
        lr_system.fit(train_vars, train_labels)
        preds = lr_system.predict_class(sel_vars)

        all_sel_vars.append(sel_vars)
        all_labels.extend(list(sel_labels))
        all_preds.extend(list(preds))
        last_lr_system = lr_system
        print()

    if last_lr_system is None:
        print("No successful training runs.")
        return

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_sel_vars_df = pd.concat(all_sel_vars, ignore_index=True)

    overall_acc = np.mean(all_preds == all_labels)
    print(f"Overall Accuracy: {overall_acc:.3f}\n")

    # --- Confusion matrix ---
    if not cfg.eval.is_multiclass:
        activity_1, activity_2 = cfg.eval.activity_pair
        display_labels = [activity_2, activity_1]  # 0 → H2, 1 → H1
    else:
        inv_map = {v: k for k, v in cfg.eval.expert_cluster_map.items()}
        display_labels = [inv_map.get(i, str(i)) for i in sorted(np.unique(all_labels))]

    plot_confusion_matrix(all_labels, all_preds, display_labels)

    # --- SHAP ---
    if cfg.scorer.name != "CatBoost":
        print(f"SHAP only implemented for CatBoost (current scorer: {cfg.scorer.name}). Skipping.")
        return

    shap_values, feature_names = get_shap_values(cfg, last_lr_system, all_sel_vars_df)
    plot_shap(shap_values, feature_names)


@hydra.main(config_path="./conf", config_name="config_main", version_base=None)
def main(cfg):
    train_select_analyse(cfg)


if __name__ == "__main__":
    main()
