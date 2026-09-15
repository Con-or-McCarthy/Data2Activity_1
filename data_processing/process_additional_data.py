"""
Process the two additional iPhone trace datasets into the same column format as
fared_min.csv.

Reads:
  <SOURCE_ROOT>/Dataset_1/ppn_*/<phone>/  (sqlite trace files per iPhone per subject)
  <SOURCE_ROOT>/Dataset_2/ppn_*/<phone>/
  data/Dataset_1/master_labels.csv        (trial labels transcribed from the protocol sheets)
  data/Dataset_2/master_labels.csv

Writes:
  data/Dataset_1/clean/dataset_1_processed.csv
  data/Dataset_2/clean/dataset_2_processed.csv

Each protocol trial has an explicit start and stop time, so activity intervals
are taken directly from master_labels.csv.  Both READMEs state the subject stood
still between trials, so the gaps between consecutive trials are labelled
'standing', trimmed by GAP_TRIM at each end so the transition into and out of a
trial does not leak into the standing class.

A label row with a start but no stop (the experimenter did not record one) is a
"blackout": it gets no activity label, and the surrounding gap is not labelled
standing either, since an untimed activity happened inside it.

SQLite timestamps are Apple epoch (seconds since 2001-01-01 UTC).  Dataset_1 was
recorded in June (CEST, UTC+2) and Dataset_2 in January (CET, UTC+1); both
offsets were verified against the protocol times before being hardcoded.
"""
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path("/home/conor/my_PhD/Datasets/additional_trace_datasets")
FARED_MIN_PATH = REPO_DIR / "data" / "NFI_FARED" / "clean" / "fared_min.csv"

APPLE_EPOCH = 978307200  # seconds between 1970-01-01 and 2001-01-01 (UTC)
GAP_TRIM = pd.Timedelta(seconds=30)  # trimmed off each end of a between-trial gap
BLACKOUT_PAD = pd.Timedelta(minutes=3)  # window excluded around a start-only trial

sys.path.insert(0, str(REPO_DIR))
from data_processing.process_data import column_cleaner, clip_to_freq, differentiate_values, combine_dfs

warnings.filterwarnings("ignore", category=FutureWarning)

# UTC offset of the local times written on the protocol sheets, per dataset.
TZ_OFFSET = {
    "Dataset_1": pd.Timedelta(hours=2),  # June 2025, CEST
    "Dataset_2": pd.Timedelta(hours=1),  # January 2025, CET
}

# Source folder name -> phone type string, in the same style as the AUAS dataset.
PHONE_TYPES = {
    "iPhone15_Pro_iOS17.5.1": "iPhone_15_Pro_iOS_17.5.1",
    "iPhone13_Pro_iOS15.2": "iPhone_13_Pro_iOS_15.2",
    "iPhone8_iOS16.7.4": "iPhone_8_iOS_16.7.4",
    "iPhone12_iOS15.2.1": "iPhone_12_iOS_15.2.1",
    "iPhone12_Pro_iOS16.6.1": "iPhone_12_Pro_iOS_16.6.1",
    "iPhone12_Mini_iOS14.4.2": "iPhone_12_Mini_iOS_14.4.2",
}

# Carrying location per (subject, phone folder), from the dataset READMEs and the
# protocol sheets.  Mapped onto the NFI_FARED vocabulary: trouser pocket front ->
# frontpocket, trouser pocket back -> backpocket, jacket pocket -> breastpocket.
CARRY_LOCATIONS = {
    ("D1_1", "iPhone15_Pro_iOS17.5.1"): "breastpocket",
    ("D1_1", "iPhone13_Pro_iOS15.2"): "backpocket",
    ("D1_1", "iPhone8_iOS16.7.4"): "hand",
    ("D1_2", "iPhone15_Pro_iOS17.5.1"): "backpocket",
    ("D1_2", "iPhone13_Pro_iOS15.2"): "frontpocket",
    ("D1_2", "iPhone8_iOS16.7.4"): "backpocket",
    ("D1_3", "iPhone15_Pro_iOS17.5.1"): "hand",
    ("D1_3", "iPhone13_Pro_iOS15.2"): "breastpocket",
    ("D1_3", "iPhone8_iOS16.7.4"): "frontpocket",
    ("D1_4", "iPhone15_Pro_iOS17.5.1"): "frontpocket",
    ("D1_4", "iPhone13_Pro_iOS15.2"): "hand",
    ("D1_4", "iPhone8_iOS16.7.4"): "breastpocket",
    ("D2_1a", "iPhone12_iOS15.2.1"): "frontpocket",
    ("D2_1a", "iPhone12_Pro_iOS16.6.1"): "frontpocket",
    ("D2_1a", "iPhone12_Mini_iOS14.4.2"): "breastpocket",
    ("D2_1b", "iPhone12_iOS15.2.1"): "frontpocket",
    ("D2_1b", "iPhone12_Pro_iOS16.6.1"): "breastpocket",
    ("D2_1b", "iPhone12_Mini_iOS14.4.2"): "backpocket",
    ("D2_2", "iPhone12_iOS15.2.1"): "frontpocket",
    ("D2_2", "iPhone12_Pro_iOS16.6.1"): "breastpocket",
    ("D2_2", "iPhone12_Mini_iOS14.4.2"): "hand",
    ("D2_3", "iPhone12_iOS15.2.1"): "hand",
    ("D2_3", "iPhone12_Pro_iOS16.6.1"): "frontpocket",
    ("D2_3", "iPhone12_Mini_iOS14.4.2"): "breastpocket",
}

# Source folder name -> subject id used in master_labels.csv.
SUBJECT_IDS = {
    ("Dataset_1", "ppn_1"): "D1_1",
    ("Dataset_1", "ppn_2"): "D1_2",
    ("Dataset_1", "ppn_3"): "D1_3",
    ("Dataset_1", "ppn_4"): "D1_4",
    ("Dataset_2", "ppn_1a"): "D2_1a",
    ("Dataset_2", "ppn_1b"): "D2_1b",
    ("Dataset_2", "ppn_2"): "D2_2",
    ("Dataset_2", "ppn_3"): "D2_3",
}


def apple_epoch_to_local(series: pd.Series, tz_offset: pd.Timedelta) -> pd.Series:
    """Convert Apple epoch seconds to naive local datetimes."""
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s + APPLE_EPOCH, unit="s", errors="coerce") + tz_offset


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    # The source tree is read-only, so the databases have to be opened immutably.
    con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def read_health_samples(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        q = """
        SELECT samples.start_date, samples.end_date, samples.data_type,
               samples.data_id, quantity_samples.quantity AS value
        FROM samples
        LEFT JOIN quantity_samples ON samples.data_id = quantity_samples.data_id
        """
        return pd.read_sql_query(q, con)
    finally:
        con.close()


def load_iphone_dfs(iphone_dir: Path, tz_offset: pd.Timedelta) -> list:
    """
    Load the feature DataFrames from an iPhone directory, in the order expected
    by combine_dfs.  Cache.sqlite is absent for every iPhone 12 Mini, so the
    location DataFrame is simply omitted when the file is missing.
    """
    cache_enc = iphone_dir / "cache_encryptedC.db"
    healthdb = iphone_dir / "healthdb_secure.sqlite"
    cache_sqlite = iphone_dir / "Cache.sqlite"

    # cache_encryptedC.db: motion, activity energy, step count
    df_motion = read_table(cache_enc, "MotionStateHistory")
    df_natalie = read_table(cache_enc, "NatalieHistory")
    df_steps = read_table(cache_enc, "StepCountHistory")
    for df in [df_motion, df_natalie, df_steps]:
        df["startTime (local time)"] = apple_epoch_to_local(df["startTime"], tz_offset)

    # healthdb_secure.sqlite: HealthKit quantities
    df_health = read_health_samples(healthdb)
    df_health["start_date"] = apple_epoch_to_local(df_health["start_date"], tz_offset)
    df_health["end_date"] = apple_epoch_to_local(df_health["end_date"], tz_offset)

    def health_subset(data_type: int, value_col: str) -> pd.DataFrame:
        sub = df_health[df_health["data_type"] == data_type][
            ["start_date", "end_date", "data_type", "data_id", "value"]
        ].copy()
        sub.rename(
            columns={
                "start_date": "startTime (local time)",
                "end_date": "end_date (local time)",
                "value": value_col,
            },
            inplace=True,
        )
        return sub

    df_h_steps = health_subset(7, "steps")     # HKQuantityTypeIdentifierStepCount
    df_h_dist = health_subset(8, "distance")   # HKQuantityTypeIdentifierDistanceWalkingRunning
    df_h_floors = health_subset(12, "floors")  # HKQuantityTypeIdentifierFlightsClimbed

    df_list = [df_steps, df_natalie, df_motion]
    if cache_sqlite.exists():
        # Cache.sqlite: ZRTCLLOCATIONMO (GPS/location data)
        df_cache = read_table(cache_sqlite, "ZRTCLLOCATIONMO")
        df_cache["ZTIMESTAMP"] = apple_epoch_to_local(df_cache["ZTIMESTAMP"], tz_offset)
        df_list.append(df_cache)
    df_list += [df_h_steps, df_h_dist, df_h_floors]
    return df_list


def build_intervals(labels_df: pd.DataFrame, subject: str) -> list:
    """
    Return sorted (start, end, activity) tuples for one subject: the labelled
    trials, plus a trimmed 'standing' interval for each gap between consecutive
    trials.  Gaps touching a start-only trial are skipped.
    """
    rows = labels_df[labels_df["META_test_subject"] == subject].copy()
    if rows.empty:
        return []

    date = pd.to_datetime(rows["META_exp_date"].iloc[0], format="%d/%m/%Y").strftime("%Y-%m-%d")

    def stamp(t):
        return pd.NaT if pd.isna(t) else pd.Timestamp(f"{date} {str(t).strip()}")

    rows["start_dt"] = rows["starttime"].apply(stamp)
    rows["stop_dt"] = rows["stoptime"].apply(stamp)
    rows = rows.sort_values("start_dt").reset_index(drop=True)

    intervals = []
    for _, row in rows.iterrows():
        if pd.notna(row["stop_dt"]) and pd.notna(row["META_label_activity"]):
            intervals.append((row["start_dt"], row["stop_dt"], row["META_label_activity"]))

    # 'standing' in the gap between each pair of consecutive trials
    for i in range(len(rows) - 1):
        prev_stop, next_start = rows.loc[i, "stop_dt"], rows.loc[i + 1, "start_dt"]
        if pd.isna(prev_stop):
            # start-only trial: its (unknown) duration falls inside this gap
            continue
        if pd.isna(rows.loc[i + 1, "stop_dt"]):
            # the next trial is start-only; keep clear of it
            next_start = next_start - BLACKOUT_PAD
        gap_start, gap_end = prev_stop + GAP_TRIM, next_start - GAP_TRIM
        if gap_start < gap_end:
            intervals.append((gap_start, gap_end, "standing"))

    return sorted(intervals)


def add_meta(df: pd.DataFrame, carrying_location, telephone_type, test_subject, experiment) -> pd.DataFrame:
    df = df.copy()
    df["META_carrying_location"] = carrying_location
    df["META_telephone_type"] = telephone_type
    df["META_test_subject"] = test_subject
    df["META_experiment"] = experiment
    df["META_label_activity"] = "unknown"
    return df


def assign_labels(df: pd.DataFrame, intervals: list) -> pd.DataFrame:
    """Label each row by the interval its startTime(localtime) falls in; None outside all."""
    df = df.copy()
    labels = []
    for ts in df["startTime(localtime)"]:
        label = None
        if pd.notna(ts):
            for start, end, activity in intervals:
                if start <= ts < end:
                    label = activity
                    break
        labels.append(label)
    df["META_label_activity"] = labels
    return df


def process_iphone(iphone_dir: Path, intervals: list, subject: str, dataset: str, freq: str) -> pd.DataFrame:
    """Load one iPhone directory, run the standard pipeline, and keep labelled rows."""
    phone_folder = iphone_dir.name
    carrying_location = CARRY_LOCATIONS[(subject, phone_folder)]

    meta = dict(
        carrying_location=carrying_location,
        telephone_type=PHONE_TYPES[phone_folder],
        test_subject=subject,
        experiment=dataset,
    )
    df_list = [add_meta(df, **meta) for df in load_iphone_dfs(iphone_dir, TZ_OFFSET[dataset])]
    df_list = [column_cleaner(df) for df in df_list]
    df_list = [clip_to_freq(df, freq=freq) for df in df_list]
    df_list = [differentiate_values(df) for df in df_list]

    combined = combine_dfs(df_list, freq=freq)
    if combined.empty:
        return combined

    combined = assign_labels(combined, intervals)
    return combined[combined["META_label_activity"].notna()].copy()


def process_dataset(dataset: str, freq: str) -> pd.DataFrame:
    labels_path = REPO_DIR / "data" / dataset / "master_labels.csv"
    labels_df = pd.read_csv(labels_path, dtype=str)

    results = []
    for subject_dir in sorted((SOURCE_ROOT / dataset).iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = SUBJECT_IDS[(dataset, subject_dir.name)]
        intervals = build_intervals(labels_df, subject)
        if not intervals:
            print(f"  {subject}: no labels, skipping")
            continue

        for iphone_dir in sorted(subject_dir.iterdir()):
            if not iphone_dir.is_dir():
                continue
            print(f"  {subject} / {iphone_dir.name} ...")
            result = process_iphone(iphone_dir, intervals, subject, dataset, freq)
            if not result.empty:
                results.append(result)
                print(f"    -> {len(result)} labelled rows")
            else:
                print("    -> no labelled rows")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def main(freq: str = "min"):
    fared_cols = pd.read_csv(FARED_MIN_PATH, nrows=1).columns.tolist()

    for dataset in ["Dataset_1", "Dataset_2"]:
        print(f"Processing {dataset} ...")
        out = process_dataset(dataset, freq)
        if out.empty:
            print(f"No data processed for {dataset}.")
            continue

        # Align output to fared_min.csv column order; add missing columns as NaN
        for col in fared_cols:
            if col not in out.columns:
                out[col] = np.nan
        extra_cols = [c for c in out.columns if c not in fared_cols]
        out = out[[c for c in fared_cols if c in out.columns] + extra_cols]

        output_path = REPO_DIR / "data" / dataset / "clean" / f"{dataset.lower()}_processed.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, na_rep="nan")
        print(f"Saved {len(out)} rows to {output_path}\n")


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "min"
    main(freq=freq)
