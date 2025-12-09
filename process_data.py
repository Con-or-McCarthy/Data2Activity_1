"""
Script to process and clean NFI_FARED dataset.
Run `bash download_original_data.sh` to download original data to the appropriate folder.
Script defaults to cleaning data to 1-minute frequency, but can be adjusted by passing a frequency argument when running the script e.g. `python process_data.py 30S` for 30-second frequency.
Results in the paper are based on 1-minute frequency.

"""
import pickle
import pandas as pd
import sys

from pathlib import Path
from tqdm import tqdm

REPO_DIR = Path('.')
PATH_TO_FOLDER = REPO_DIR / 'data' / 'NFI_FARED' / 'original' # Location where NFI_FARED is stored 
SAVE_FOLDER = REPO_DIR / 'data' / 'NFI_FARED' / 'clean' 

def load_files():
    with open(PATH_TO_FOLDER / "df_dict_stepcounthistory.pkl", 'rb') as picklefile:
        df_dict_stepcounthistory = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_natalie.pkl", 'rb') as picklefile:
        df_dict_natalie = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_motionstate.pkl", 'rb') as picklefile:
        df_dict_motionstate = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_Cache.pkl", 'rb') as picklefile:
        df_dict_Cache = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_healthdb_steps.pkl", 'rb') as picklefile:
        df_dict_healthdb_steps = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_healthdb_distance.pkl", 'rb') as picklefile:
        df_dict_healthdb_distance = pickle.load(picklefile)
    with open(PATH_TO_FOLDER / "df_dict_healthdb_floors.pkl", 'rb') as picklefile:
        df_dict_healthdb_floors = pickle.load(picklefile)

    df_dict_healthdb_steps.rename(columns={"start_date (local time)": "startTime (local time)"}, inplace=True)
    df_dict_healthdb_distance.rename(columns={"start_date (local time)": "startTime (local time)"}, inplace=True)
    df_dict_healthdb_floors.rename(columns={"start_date (local time)": "startTime (local time)"}, inplace=True)

    return df_dict_stepcounthistory, df_dict_natalie, df_dict_motionstate, df_dict_Cache, df_dict_healthdb_steps, df_dict_healthdb_distance, df_dict_healthdb_floors

def column_cleaner(df):
    """
    Removes spaces from column names and renames columns to a more consistent format.
    """
    # Clean names of columns
    df.columns = df.columns.str.replace(' ', '')
    return df

def differentiate_values(df):
    """
    Gets the difference between consecutive values in the dataframe.
    If the time difference between two consecutive rows is greater than 2 minutes,
    the difference is set to 0.
    
    """
    # Columns to differentiate
    all_diff_cols = ['floorsAscended', 'floorsDescended', 'rawdistance', 'health_distance', 'pushCount', 'count']
    diff_cols = [col for col in all_diff_cols if col in df.columns]
    
    if len(diff_cols) == 0:
        return df
    
    # Identifier columns
    id_cols = ['META_carrying_location', 'META_telephone_type', 'META_test_subject', 'META_experiment']
    # Filter to only include identifier columns that exist in the dataframe
    id_cols = [col for col in id_cols if col in df.columns]
    
    # Make a copy of the original dataframe to store the results
    result_df = df.copy()
    # Group by the identifier columns
    groups = df.groupby(id_cols)
    
    # For each group, calculate the differences
    for group_name, group_df in groups:
        # Get the indices for this group
        indices = group_df.index
        # For each column to differentiate
        for col in diff_cols:
            # Calculate time differences in minutes within this group
            time_diff = group_df['startTime(localtime)'].diff().dt.total_seconds() / 60
            # Calculate the regular diff for the column within this group
            value_diff = group_df[col].diff()
            # Where time difference exceeds 2 minutes, set the diff to 0
            value_diff[time_diff > 2] = 0
            # Fill any remaining NaN values with 0 (like the first row)
            # Assign back to the result dataframe using the original indices
            result_df.loc[indices, col] = value_diff.fillna(0).astype("float")
    
    return result_df

def clip_to_freq(df, freq):
    """
    For rows with 'startTime(localtime)', 'end_date(localtime)', and 'steps':
    1. Check if difference between start and end time is less than 1.2 * frequency minutes
    2. If condition is met, change start time to the midpoint between start and end time
    3. Otherwise, remove the row
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with time columns
    freq : str
        Frequency string for determining the threshold (e.g., '1min', '30S')
        
    Returns:
    --------
    DataFrame
        Processed dataframe
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if required columns exist
    required_cols = ['startTime(localtime)', 'end_date(localtime)']
    if not all(col in result_df.columns for col in required_cols):
        return result_df  # Return original if not all required columns exist

    if 'distance'  in result_df.columns:
        result_df.rename(columns={'distance': 'health_distance'}, inplace=True)
    
    # Calculate time difference in minutes
    result_df['time_diff'] = (result_df['end_date(localtime)'] - result_df['startTime(localtime)']).dt.total_seconds() / 60
    
    # Convert freq to minutes (assuming freq is in the format like '1min' or '30S')
    freq_in_minutes = 1  # Default to 1 minute if we can't parse freq
    try:
        if 'min' in freq:
            freq_in_minutes = float(freq.replace('min', ''))
        elif 'S' in freq:
            freq_in_minutes = float(freq.replace('S', '')) / 60
        elif 'H' in freq or 'h' in freq:
            freq_in_minutes = float(freq.replace('H', '').replace('h', '')) * 60
    except (ValueError, TypeError):
        pass
    
    # Filter rows based on time difference condition
    keep_rows = result_df['time_diff'] < 1.2 * freq_in_minutes
    
    # For rows that meet the condition, update starttimetext to the midpoint
    for idx in result_df[keep_rows].index:
        start_time = result_df.loc[idx, 'startTime(localtime)']
        end_time = result_df.loc[idx, 'end_date(localtime)']
        
        # Calculate midpoint
        midpoint = start_time + (end_time - start_time) / 2
        
        # Update start time to midpoint
        result_df.loc[idx, 'startTime(localtime)'] = midpoint
    
    # Keep only rows that meet condition and drop the temporary column
    result_df = result_df[keep_rows].drop(columns=['time_diff'])
    
    return result_df


def combine_dfs(df_list, freq="min"):
    """
    Combines multiple dataframes and aggregates duplicate entries.
    
    Parameters:
    -----------
    df_list : list of DataFrames
        List of dataframes to combine
    freq : str, default "min"
        Frequency to round timestamps to        
    Returns:
    --------
    DataFrame
        Combined dataframe with aggregated values
    """

    sum_cols = ['activeTime', 'count', 'distance', 'floorsAscended', 'floorsDescended', 'pushCount', 'rawdistance', 'basalNatalies', 'natalies', 'steps', 'health_distance']
    maj_cols = ['workoutType', 'activityType', 'isVehicular', 'isMoving', 'mounted', 'turn', 'type', 'vehicularFlagsData', 'ZSIGNALENVIRONMENTTYPE', "ZTYPE"]
    avg_cols = ["pace", 'mets', 'confidence', 'mountedConfidence', 'ZALTITUDE', 'ZCOURSE', 'ZHORIZONTALACCURACY', 'ZLATITUDE', 'ZLONGITUDE', 'ZSPEED', 'ZVERTICALACCURACY', 'Z_ENT', 'Z_OPT', 'Z_PK']
    
    # Define majority function that ignores NaN values
    def majority_count(x):
        # Filter out NaN values
        x = x.dropna()
        if len(x) == 0:
            return pd.NA
        # Return the most common value
        return x.value_counts().index[0]
    
    # Define mean function that ignores NaN values
    def mean_ignoring_nan(x):
        # First check if we can convert to numeric
        try:
            # Convert to numeric, coerce non-convertible values to NaN
            numeric_x = pd.to_numeric(x, errors='coerce')
            # Drop NaN values
            numeric_x = numeric_x.dropna()
            if len(numeric_x) == 0:
                return pd.NA
            return numeric_x.mean()
        except:
            # If conversion fails or for other errors, use the majority
            return majority_count(x)
    
    # Define sum function that ignores NaN values
    def sum_ignoring_nan(x):
        try:
            # Convert to numeric, coerce non-convertible values to NaN
            numeric_x = pd.to_numeric(x, errors='coerce')
            # Drop NaN values
            numeric_x = numeric_x.dropna()
            if len(numeric_x) == 0:
                return pd.NA
            return numeric_x.sum()
        except:
            # If conversion fails or for other errors, return NaN
            return pd.NA
    
        # Custom function to safely combine values during merge
    def safe_combine(values, method='majority'):
        # Filter out None and NaN
        filtered_values = [v for v in values if v is not None and not pd.isna(v)]
        
        if not filtered_values:
            return pd.NA
        
        if method == 'sum':
            try:
                # Try to convert all to numeric
                numeric_values = [pd.to_numeric(v) for v in filtered_values]
                return sum(numeric_values)
            except:
                return pd.NA
        elif method == 'mean':
            try:
                # Try to convert all to numeric
                numeric_values = [pd.to_numeric(v) for v in filtered_values]
                return sum(numeric_values) / len(numeric_values)
            except:
                return pd.NA
        else:  # majority
            # Count occurrences
            from collections import Counter
            counts = Counter(filtered_values)
            if not counts:
                return pd.NA
            # Return most common
            return counts.most_common(1)[0][0]

    # First: aggregate each dataframe individually to handle duplicates within each source
    processed_dfs = []
    for df in tqdm(df_list):
        df = df.copy()  # Avoid modifying the original dataframes
        
        # Handle ZTIMESTAMP if it exists
        if 'ZTIMESTAMP' in df.columns:
            df["startTime(localtime)"] = df["ZTIMESTAMP"].dt.round(freq)
        else:
            # Make sure startTime (local time) exists and round it
            if 'startTime(localtime)' in df.columns:
                df['startTime(localtime)'] = df['startTime(localtime)'].dt.round(freq)
        
        # Create aggregation function dictionary for all columns
        agg_funcs = {}
        for col in df.columns:
            if col in ["startTime(localtime)", "META_carrying_location", "META_telephone_type", "META_test_subject", "META_label_activity"]:
                # Skip the key columns as they're already used for grouping
                continue
            elif col in sum_cols:
                agg_funcs[col] = sum_ignoring_nan
            elif col in maj_cols:
                agg_funcs[col] = majority_count
            elif col in avg_cols:
                agg_funcs[col] = mean_ignoring_nan
            else:
                # Default to majority for any unlisted columns
                agg_funcs[col] = majority_count
        
        # Only do groupby if there are enough rows to potentially have duplicates
        if len(df) > 1 and agg_funcs:
            # Group by the key columns and apply the aggregation functions
            df = df.groupby(
                ["startTime(localtime)", "META_carrying_location", "META_telephone_type", "META_test_subject", "META_label_activity"]
            ).agg(agg_funcs).reset_index()
            
        processed_dfs.append(df)
    
    # Second: merge all the pre-aggregated dataframes
    if not processed_dfs:
        return pd.DataFrame()  # Return empty dataframe if no input
        
    result = processed_dfs[0]
    
    # Only proceed with merges if there's more than one dataframe
    for i, df in enumerate(processed_dfs[1:], 1):
        # For columns that exist in both dataframes, we need to handle them specially
        overlap_cols = set(result.columns).intersection(set(df.columns)) - set(
            ["startTime(localtime)", "META_carrying_location", "META_telephone_type", "META_test_subject", "META_label_activity"]
        )
        
        # Perform the merge
        result = pd.merge(
            result, df, 
            on=["startTime(localtime)", "META_carrying_location", "META_telephone_type", "META_test_subject", "META_label_activity"], 
            how="outer",
            suffixes=("", f"_{i}")
        )
        
        # After merging, we need to handle the columns that were in both dataframes
        for col in overlap_cols:
            col_suffix = f"{col}_{i}"
            if col_suffix in result.columns:
                # Choose appropriate aggregation based on column type
                if col in sum_cols:
                    # For sum columns, add the values (ignoring NaNs)
                    result[col] = result.apply(
                        lambda row: safe_combine([row[col], row[col_suffix]], method='sum'), 
                        axis=1
                    )
                elif col in avg_cols:
                    # For average columns, take the mean (ignoring NaNs)
                    result[col] = result.apply(
                        lambda row: safe_combine([row[col], row[col_suffix]], method='mean'), 
                        axis=1
                    )
                else:
                    # For majority columns, take the non-NaN value or the first value
                    result[col] = result.apply(
                        lambda row: safe_combine([row[col], row[col_suffix]], method='majority'), 
                        axis=1
                    )
                
                # Drop the duplicate column
                result = result.drop(columns=[col_suffix])
    
    return result


def save_dfs(df, freq):
    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_FOLDER / f"fared_{freq}.csv", index=False, na_rep='nan')
    
def main(freq='min'):
    print("loading files..")
    df_dict_stepcounthistory, df_dict_natalie, df_dict_motionstate, df_dict_Cache, df_dict_healthdb_steps, df_dict_healthdb_distance, df_dict_healthdb_floors = load_files()
    print("processing dataframes..")
    df_list = [df_dict_stepcounthistory, df_dict_natalie, df_dict_motionstate, df_dict_Cache, df_dict_healthdb_steps, df_dict_healthdb_distance, df_dict_healthdb_floors]
    print('cleaning column names...')
    for i in tqdm(range(len(df_list))):
        df_list[i] = column_cleaner(df_list[i])
    print("Clipping health rows...")
    for i in tqdm(range(len(df_list))):
        df_list[i] = clip_to_freq(df_list[i], freq='min')
    print("Getting row differences...")
    for i in tqdm(range(len(df_list))):
        df_list[i] = differentiate_values(df_list[i])
    print('combining dfs...')
    big_df = combine_dfs(df_list, freq=freq)
    print("saving...")
    save_dfs(big_df, freq)
    print("finished!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        freq = sys.argv[1]
    else:
        freq = 'min'

    print(f"Frequency set to: {freq}")
    print("Starting data processing...")
    main(freq=freq)