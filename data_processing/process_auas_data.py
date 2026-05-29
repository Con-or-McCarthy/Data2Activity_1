"""
Process AUAS iPhone trace data into the same column format as fared_min.csv.

Reads:
  data/AUAS/original/  (sqlite trace files per iPhone per experiment)
  data/AUAS/master_labels.csv  (activity labels with timestamps)

Writes:
  data/AUAS/clean/auas_processed.csv

Activity labels are assigned by matching each 1-minute-rounded timestamp
against activity intervals derived from master_labels.csv.  starttime_video
combined with META_exp_date defines when each activity starts; the next
activity start is the end, except for the last activity which gets +30s.

SQLite timestamps are Apple epoch (seconds since 2001-01-01 UTC) and are
converted to CET (UTC+1) to match the local times in master_labels.csv.
"""
import re
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent
AUAS_ORIGINAL = REPO_DIR / "data" / "AUAS" / "original"
MASTER_LABELS_PATH = REPO_DIR / "data" / "AUAS" / "master_labels.csv"
OUTPUT_PATH = REPO_DIR / "data" / "AUAS" / "clean" / "auas_processed.csv"
FARED_MIN_PATH = REPO_DIR / "data" / "NFI_FARED" / "clean" / "fared_min.csv"

APPLE_EPOCH = 978307200  # seconds between 1970-01-01 and 2001-01-01 (UTC)
TZ_OFFSET = pd.Timedelta(hours=1)  # CET = UTC+1 (November/December 2024)

sys.path.insert(0, str(REPO_DIR))
from data_processing.process_data import column_cleaner, clip_to_freq, differentiate_values, combine_dfs

warnings.filterwarnings("ignore", category=FutureWarning)


def apple_epoch_to_cet(series: pd.Series) -> pd.Series:
    """Convert Apple epoch seconds to naive CET datetimes."""
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s + APPLE_EPOCH, unit="s", errors="coerce") + TZ_OFFSET


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def read_health_samples(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
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


def parse_experiment_dir(exp_dir: Path) -> dict:
    """Extract exp_num, date_str (DD/MM/YYYY), person_id from experiment folder name."""
    m = re.search(
        r'[Ee]xperiment[\s_]*(\d+\.\d+)[_\s](\d+)[_\s](\d+)[_\s](\d+)[_\s][Pp]ersoon[Ii][Dd](\d+)',
        exp_dir.name,
    )
    if not m:
        raise ValueError(f"Cannot parse experiment folder: {exp_dir.name}")
    day, month, year = m.group(2), m.group(3), m.group(4)
    return {
        "exp_num": m.group(1),
        "date_str": f"{day}/{month}/{year}",
        "person_id": int(m.group(5)),
    }


def parse_phone_type(iphone_dir: Path) -> str:
    """Normalize iPhone folder name to a consistent phone type string including phone number."""
    m = re.search(r'[Ii]phone[\s_]*(\w+)[\s_]+[Ii][Oo][Ss][\s_]*([\d.]+)[\s_]+nr(\d+)', iphone_dir.name)
    if m:
        return f"iPhone_{m.group(1)}_iOS_{m.group(2)}_nr{m.group(3)}"
    m = re.search(r'[Ii]phone[\s_]*(\w+)[\s_]+[Ii][Oo][Ss][\s_]*([\d.]+)', iphone_dir.name)
    if m:
        return f"iPhone_{m.group(1)}_iOS_{m.group(2)}"
    return iphone_dir.name


def get_label_intervals(labels_df: pd.DataFrame, exp_num: str, date_str: str, person_id: int) -> list:
    """Return sorted list of (start_dt, end_dt, activity_label) tuples for one experiment."""
    mask = (
        labels_df["META_experiment"].apply(lambda x: f"{float(x):.1f}") == f"{float(exp_num):.1f}"
    ) & (
        labels_df["META_exp_date"].str.strip() == date_str
    ) & (
        labels_df["META_person_ID"] == person_id
    )
    rows = labels_df[mask].copy()
    if rows.empty:
        return []

    exp_date = pd.to_datetime(date_str, format="%d/%m/%Y").strftime("%Y-%m-%d")
    rows["start_dt"] = rows["starttime_video"].str.strip().apply(
        lambda t: pd.Timestamp(f"{exp_date} {t}")
    )
    rows = rows.sort_values("start_dt").reset_index(drop=True)

    intervals = []
    for i, row in rows.iterrows():
        start = row["start_dt"]
        end = rows.loc[i + 1, "start_dt"] if i < len(rows) - 1 else start + pd.Timedelta(seconds=30)
        intervals.append((start, end, row["META_label_activity"].strip()))
    return intervals


def load_iphone_dfs(iphone_dir: Path) -> tuple:
    """
    Load the 7 feature DataFrames from an iPhone directory.

    Returns (df_steps, df_natalie, df_motion, df_cache, df_h_steps, df_h_dist, df_h_floors)
    matching the order expected by combine_dfs in process_data.py.
    """
    def find_file(pattern: str) -> Path:
        matches = sorted(iphone_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' in {iphone_dir}")
        return matches[0]

    cache_sqlite = find_file("Cache.sqlite")
    cache_enc = find_file("cache_encryptedC.db")
    healthdb = find_file("healthdb_secure.sqlite")

    # Cache.sqlite: ZRTCLLOCATIONMO (GPS/location data)
    df_cache = read_table(cache_sqlite, "ZRTCLLOCATIONMO")
    df_cache["ZTIMESTAMP"] = apple_epoch_to_cet(df_cache["ZTIMESTAMP"])

    # cache_encryptedC.db: motion, activity energy, step count
    df_motion = read_table(cache_enc, "MotionStateHistory")
    df_natalie = read_table(cache_enc, "NatalieHistory")
    df_steps = read_table(cache_enc, "StepCountHistory")
    for df in [df_motion, df_natalie, df_steps]:
        df["startTime (local time)"] = apple_epoch_to_cet(df["startTime"])

    # healthdb_secure.sqlite: HealthKit quantities
    df_health = read_health_samples(healthdb)
    df_health["start_date"] = apple_epoch_to_cet(df_health["start_date"])
    df_health["end_date"] = apple_epoch_to_cet(df_health["end_date"])

    def health_subset(data_type: int, value_col: str) -> pd.DataFrame:
        sub = df_health[df_health["data_type"] == data_type][
            ["start_date", "end_date", "data_type", "data_id", "value"]
        ].copy()
        # Rename to match what clip_to_freq and process_data expect
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
    df_h_dist = health_subset(8, "distance")    # HKQuantityTypeIdentifierDistanceWalkingRunning
    df_h_floors = health_subset(12, "floors")   # HKQuantityTypeIdentifierFlightsClimbed

    return df_steps, df_natalie, df_motion, df_cache, df_h_steps, df_h_dist, df_h_floors


def add_meta(df: pd.DataFrame, carrying_location, telephone_type, test_subject, experiment) -> pd.DataFrame:
    df = df.copy()
    df["META_carrying_location"] = carrying_location
    df["META_telephone_type"] = telephone_type
    df["META_test_subject"] = test_subject
    df["META_experiment"] = experiment
    df["META_label_activity"] = "unknown"
    return df


def assign_labels(df: pd.DataFrame, intervals: list) -> pd.DataFrame:
    """
    Set META_label_activity on each row based on which interval
    startTime(localtime) falls in. Rows outside all intervals get None.
    """
    if "startTime(localtime)" not in df.columns:
        return df
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


def process_iphone(
    iphone_dir: Path,
    labels_df: pd.DataFrame,
    exp_meta: dict,
    phone_type: str,
    freq: str = "min",
) -> pd.DataFrame:
    """
    Process one iPhone directory:
    1. Load sqlite files
    2. Run standard processing pipeline (column_cleaner → clip_to_freq → differentiate_values → combine_dfs)
    3. Assign activity labels from master_labels.csv
    4. Drop rows outside labeled intervals
    5. Set final META columns
    """
    test_subject = f"AU_{exp_meta['person_id']}"
    experiment = exp_meta["exp_num"]

    intervals = get_label_intervals(
        labels_df, exp_meta["exp_num"], exp_meta["date_str"], exp_meta["person_id"]
    )
    if not intervals:
        print("  No labels found, skipping.")
        return pd.DataFrame()

    try:
        df_tuple = load_iphone_dfs(iphone_dir)
    except FileNotFoundError as e:
        print(f"  Missing file: {e}")
        return pd.DataFrame()

    # Use "unknown" as carrying_location placeholder so groupby in combine_dfs works
    meta = dict(
        carrying_location="unknown",
        telephone_type=phone_type,
        test_subject=test_subject,
        experiment=experiment,
    )
    # Order: stepcounthistory, natalie, motionstate, Cache, healthdb_steps, healthdb_dist, healthdb_floors
    df_list = [add_meta(df, **meta) for df in df_tuple]
    df_list = [column_cleaner(df) for df in df_list]
    df_list = [clip_to_freq(df, freq=freq) for df in df_list]
    df_list = [differentiate_values(df) for df in df_list]

    combined = combine_dfs(df_list, freq=freq)
    if combined.empty:
        return combined

    combined = assign_labels(combined, intervals)
    combined = combined[combined["META_label_activity"].notna()].copy()
    combined["META_carrying_location"] = np.nan
    return combined


def main(freq: str = "min"):
    labels_df = pd.read_csv(MASTER_LABELS_PATH)
    fared_cols = pd.read_csv(FARED_MIN_PATH, nrows=1).columns.tolist()

    all_results = []
    for exp_top_dir in sorted(AUAS_ORIGINAL.iterdir()):
        if not exp_top_dir.is_dir():
            continue
        for exp_dir in sorted(exp_top_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            iphones_dir = exp_dir / "Iphones"
            if not iphones_dir.exists():
                continue
            try:
                exp_meta = parse_experiment_dir(exp_dir)
            except ValueError as e:
                print(f"Skipping {exp_dir.name}: {e}")
                continue

            for iphone_dir in sorted(iphones_dir.iterdir()):
                if not iphone_dir.is_dir():
                    continue
                phone_type = parse_phone_type(iphone_dir)
                print(f"Processing {exp_dir.name} / {iphone_dir.name} ...")
                result = process_iphone(iphone_dir, labels_df, exp_meta, phone_type, freq=freq)
                if not result.empty:
                    all_results.append(result)
                    print(f"  → {len(result)} labeled rows")

    if not all_results:
        print("No data processed.")
        return

    out = pd.concat(all_results, ignore_index=True)

    # Align output to fared_min.csv column order; add missing columns as NaN
    for col in fared_cols:
        if col not in out.columns:
            out[col] = np.nan
    extra_cols = [c for c in out.columns if c not in fared_cols]
    out = out[[c for c in fared_cols if c in out.columns] + extra_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, na_rep="nan")
    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "min"
    main(freq=freq)
