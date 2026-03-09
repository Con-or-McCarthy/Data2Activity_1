"""
Script to train a model on the full fared_min dataset and evaluate on user-uploaded data.

User data is loaded from .pkl files in user_data/pkl_files/ (df_dict_stepcounthistory.pkl,
df_dict_natalie.pkl, df_dict_motionstate.pkl, df_dict_Cache.pkl, df_dict_healthdb_steps.pkl,
df_dict_healthdb_distance.pkl, df_dict_healthdb_floors.pkl). These are processed using the
same pipeline as process_data.py and saved to user_data/processed/user_processed_files.csv.
The analysis output is saved to user_data/output/output.csv.

Usage (with Hydra overrides):
    # basic
    python use_your_data.py eval.is_multiclass=true

    # custom pkl folder and output path (relative to repo root)
    python use_your_data.py +pkl_path="mypickles/pickles" +output_path="results/my_output.csv"

Output:
    - Binary: columns [time, \"{activity_0}/{activity_1}\", \"{activity_1}/{activity_0}\"]
      where activity_0 = cfg.eval.activity_pair[0], activity_1 = cfg.eval.activity_pair[1].
    - Multiclass: columns [time] + one column per cluster in cfg.eval.expert_cluster_choices
      (or all clusters if that is None), containing log-likelihoods for each cluster.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

import hydra

from process_data import process_user_pkl_files
from utils import (
    load_data,
    setup_scorer,
    setup_calibrator,
    CalibratedScorer,
)

# Default paths for user data
USER_PKL_FOLDER = REPO_DIR / "user_data" / "pkl_files"
USER_PROCESSED_CSV = REPO_DIR / "user_data" / "processed" / "user_processed_files.csv"
USER_OUTPUT_CSV = REPO_DIR / "user_data" / "output" / "output.csv"

# Columns from fared_min that are NOT META (user data should have these)
META_COLUMNS = [
    "META_carrying_location",
    "META_telephone_type",
    "META_test_subject",
    "META_label_activity",
    "META_experiment",
]


def get_expected_columns():
    """Get full column list from fared_min for validation."""
    ref_path = REPO_DIR / "data" / "NFI_FARED" / "clean" / "fared_min.csv"
    ref_df = pd.read_csv(ref_path, nrows=1)
    return list(ref_df.columns)


def get_user_expected_columns():
    """Columns user data must have (all non-META columns from fared_min)."""
    all_cols = get_expected_columns()
    return [c for c in all_cols if c not in META_COLUMNS]


def load_and_validate_user_data(user_path: Path, cfg) -> pd.DataFrame:
    """Load user CSV and validate it has required columns."""
    user_df = pd.read_csv(user_path)

    expected_cols = get_user_expected_columns()
    missing = [c for c in expected_cols if c not in user_df.columns]
    if missing:
        raise ValueError(
            f"User CSV missing required columns: {missing}. "
            f"Expected columns (same order as fared_min): {expected_cols[:10]}..."
        )

    # Ensure vars_to_use columns that exist in reference data are present
    ref_path = REPO_DIR / "data" / "NFI_FARED" / "clean" / "fared_min.csv"
    ref_df = pd.read_csv(ref_path, nrows=1)
    vars_to_use = [c for c in cfg.data.vars_to_use if c in ref_df.columns]
    missing_vars = [v for v in vars_to_use if v not in user_df.columns]
    if missing_vars:
        raise ValueError(f"User CSV missing feature columns: {missing_vars}")

    # Parse timestamp
    timestamp_col = "startTime(localtime)"
    if timestamp_col not in user_df.columns:
        # Try alternate
        for col in ["startTime(localtime)", "startTime (localtime)", "timestamp"]:
            if col in user_df.columns:
                timestamp_col = col
                break
        else:
            raise ValueError(f"User CSV must have a timestamp column (e.g. {timestamp_col})")

    return user_df


def add_dummy_meta_columns(user_df: pd.DataFrame) -> pd.DataFrame:
    """Add dummy META columns so user data can use the same processing pipeline."""
    user_df = user_df.copy()
    if "META_carrying_location" not in user_df.columns:
        user_df["META_carrying_location"] = "unknown"
    if "META_telephone_type" not in user_df.columns:
        user_df["META_telephone_type"] = "unknown"
    if "META_test_subject" not in user_df.columns:
        user_df["META_test_subject"] = "user01"
    if "META_label_activity" not in user_df.columns:
        user_df["META_label_activity"] = "unknown"
    if "META_experiment" not in user_df.columns:
        user_df["META_experiment"] = "user"
    return user_df


def prepare_train_data(cfg):
    """Prepare training data from full fared_min (all subjects)."""
    data = load_data(cfg)

    if cfg.data.name == "NFI":
        pp_map = {"pp11": "pp03", "pp12": "pp05", "pp15": "pp04"}
        data["META_test_subject"] = data["META_test_subject"].map(
            lambda x: pp_map.get(x, x)
        )

    if not cfg.eval.is_multiclass:
        activity_1, activity_2 = cfg.eval.activity_pair
        assert activity_1 != activity_2
        acts_to_use = [activity_1, activity_2]
        data = data[data["META_label_activity"].isin(acts_to_use)].copy()
        data["label"] = data["META_label_activity"].map(
            {activity_1: 1, activity_2: 0}
        )
    else:
        cluster_map = cfg.eval.expert_cluster_assignment
        data["label"] = data["META_label_activity"].map(cluster_map)
        if cfg.eval.expert_cluster_choices is not None:
            desired_clusters = [
                cfg.eval.expert_cluster_map[choice]
                for choice in cfg.eval.expert_cluster_choices
            ]
            data = data[data["label"].isin(desired_clusters)]

    # Filter by phone/carry if specified
    if cfg.eval.phone_types is not None:
        data = data[data["META_telephone_type"].isin(cfg.eval.phone_types)].copy()
    if cfg.eval.carry_locations is not None:
        data = data[data["META_carrying_location"].isin(cfg.eval.carry_locations)].copy()

    # Use ALL data as train (no train/select split)
    cols_to_use = [col for col in cfg.data.vars_to_use if col in data.columns]
    train_vars = data.filter(items=cols_to_use, axis=1)
    train_labels = data["label"]

    train_mean, train_std = None, None
    if cfg.eval.normalise:
        normalisable_cols = [
            col for col in cfg.data.normalisable_vars if col in train_vars.columns
        ]
        if normalisable_cols:
            train_mean = train_vars[normalisable_cols].mean()
            train_std = train_vars[normalisable_cols].std()
            train_vars = train_vars.copy()
            train_vars[normalisable_cols] = (
                train_vars[normalisable_cols] - train_mean
            ) / train_std

    if not cfg.eval.is_multiclass:
        assert sum(train_labels == 1) > 0, f"No samples of {activity_1} in training data"
        assert sum(train_labels == 0) > 0, f"No samples of {activity_2} in training data"
    else:
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)

    return train_vars, train_labels, train_mean, train_std


def prepare_test_data(user_df: pd.DataFrame, cfg, train_mean=None, train_std=None):
    """Prepare user data for prediction."""
    user_df = add_dummy_meta_columns(user_df)
    cols_to_use = [col for col in cfg.data.vars_to_use if col in user_df.columns]
    test_vars = user_df.filter(items=cols_to_use, axis=1)

    if cfg.eval.normalise and train_mean is not None and train_std is not None:
        normalisable_cols = [
            col for col in cfg.data.normalisable_vars if col in test_vars.columns
        ]
        if normalisable_cols:
            test_vars = test_vars.copy()
            test_vars[normalisable_cols] = (
                test_vars[normalisable_cols] - train_mean
            ) / train_std

    return test_vars


def get_timestamp_column(user_df: pd.DataFrame) -> str:
    """Get the timestamp column name from user data."""
    for col in ["startTime(localtime)", "startTime (localtime)", "timestamp"]:
        if col in user_df.columns:
            return col
    return "startTime(localtime)"


def run_with_config(cfg, user_path: Path, output_path: Path):
    """Train on fared_min, predict on user data, write output CSV."""
    print("Loading training data...")
    train_vars, train_labels, train_mean, train_std = prepare_train_data(cfg)

    print("Loading user data...")
    user_df = load_and_validate_user_data(user_path, cfg)
    test_vars = prepare_test_data(user_df, cfg, train_mean, train_std)

    print("Setting up scorer and calibrator...")
    scorer = setup_scorer(cfg, train_vars, train_labels)
    calibrator = setup_calibrator(cfg)
    lr_system = CalibratedScorer(cfg, scorer, calibrator)

    print("Training model on full fared_min...")
    lr_system.fit(train_vars, train_labels)

    # Build output dataframe
    timestamp_col = get_timestamp_column(user_df)
    timestamps = user_df[timestamp_col].values

    print("Computing likelihood-based outputs for user data...")
    if cfg.eval.is_multiclass:
        # Run through pipeline to populate calibrator.log_likelihoods
        _ = lr_system.predict_lr(test_vars)

        # Get underlying calibrator (unwrap ELUBbounder if present)
        calibrator = lr_system.calibrator
        if hasattr(calibrator, "first_step_calibrator"):
            calibrator = calibrator.first_step_calibrator

        # These are log-likelihoods per cluster (shape: n x K)
        log_likelihoods = calibrator.log_likelihoods

        cluster_map = cfg.eval.expert_cluster_assignment

        # Determine which cluster IDs to output and their column indices
        if cfg.eval.expert_cluster_choices is not None:
            # Use only requested expert clusters, in the specified order
            chosen_names = cfg.eval.expert_cluster_choices
            chosen_ids = [cfg.eval.expert_cluster_map[name] for name in chosen_names]
            unique_clusters = sorted(chosen_ids)
        else:
            # Use all clusters present in the expert assignment map
            unique_clusters = sorted(set(cluster_map.values()))
            # Derive human-readable names from expert_cluster_map
            id_to_name = {}
            for name, cid in cfg.eval.expert_cluster_map.items():
                id_to_name[cid] = name
            chosen_names = [id_to_name.get(cid, f"cluster_{cid}") for cid in unique_clusters]

        # Columns in log_likelihoods follow sorted cluster IDs (LogReg multiclass)
        cluster_to_col = {cid: idx for idx, cid in enumerate(unique_clusters)}

        data_cols = {"timestamp": timestamps}
        for name in chosen_names:
            if cfg.eval.expert_cluster_choices is not None:
                # Map from expert cluster name to its numeric id
                cid = cfg.eval.expert_cluster_map[name]
            else:
                # In this branch, chosen_names are aligned with unique_clusters order
                idx = chosen_names.index(name)
                cid = unique_clusters[idx]

            col_idx = cluster_to_col[cid]
            # Convert log-likelihoods to likelihoods
            data_cols[name] = np.exp(log_likelihoods[:, col_idx])

        out_df = pd.DataFrame(data_cols)
    else:
        # Binary case: output both LR directions
        lrs_raw = lr_system.predict_lr(test_vars)
        lrs = np.asarray(lrs_raw)
        # For some calibrators, predict_lr may return [LR, extra_info]
        if lrs.ndim == 2:
            lrs = lrs[:, 0]

        act0, act1 = cfg.eval.activity_pair
        lr_01 = lrs  # activity_0 / activity_1
        # Avoid division by very small numbers
        lr_10 = 1.0 / np.maximum(lr_01, 1e-12)

        out_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                f"{act0}/{act1}": lr_01,
                f"{act1}/{act0}": lr_10,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")


@hydra.main(config_path="./conf", config_name="config_main", version_base=None)
def main(cfg):
    """
    Hydra entrypoint.

    Use standard Hydra command line overrides, e.g.:
        python use_your_data.py eval.is_multiclass=true
    """
    # Optional CLI overrides for pkl folder and output path, e.g.:
    #   +pkl_path="mypickles/pickles"
    #   +output_path="results/my_output.csv"
    if "pkl_path" in cfg and cfg.pkl_path is not None:
        pkl_folder = (REPO_DIR / Path(cfg.pkl_path)).resolve()
    else:
        pkl_folder = USER_PKL_FOLDER
    if not pkl_folder.exists():
        raise FileNotFoundError(
            f"User pkl folder not found: {pkl_folder}. "
            f"Please create it and add the required .pkl files."
        )

    # Step 1: Process user pkl files into CSV
    processed_csv = USER_PROCESSED_CSV
    process_user_pkl_files(pkl_folder, processed_csv)

    # Step 2: Run analysis on processed user data
    if "output_path" in cfg and cfg.output_path is not None:
        output_path = (REPO_DIR / Path(cfg.output_path)).resolve()
    else:
        output_path = USER_OUTPUT_CSV
    run_with_config(cfg, processed_csv, output_path)


if __name__ == "__main__":
    main()
