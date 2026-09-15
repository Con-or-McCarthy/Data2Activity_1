import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

AUAS_PATH = Path("data/AUAS/clean/auas_processed.csv")
NFI_PATH = Path("data/NFI_FARED/clean/fared_min.csv")

META_COLS = {
    "startTime(localtime)", "META_carrying_location", "META_telephone_type",
    "META_test_subject", "META_label_activity", "META_experiment", "id",
    "startTime(epoch)", "end_date(localtime)", "data_id", "data_type",
    "ZTIMESTAMP", "starttimeepoch", "timestamp", "startTime",
}


def load_activity(path: Path, activity: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["META_label_activity"] == activity]


def get_numeric_feature_cols(auas: pd.DataFrame, nfi: pd.DataFrame) -> list[str]:
    shared = set(auas.columns) & set(nfi.columns) - META_COLS
    cols = []
    for c in sorted(shared):
        if pd.api.types.is_numeric_dtype(auas[c]) and pd.api.types.is_numeric_dtype(nfi[c]):
            if auas[c].notna().any() or nfi[c].notna().any():
                cols.append(c)
    return cols


def plot_distributions(auas: pd.DataFrame, nfi: pd.DataFrame, feature_cols: list[str], activity: str):
    n = len(feature_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        a_vals = auas[col].dropna()
        n_vals = nfi[col].dropna()

        if a_vals.empty and n_vals.empty:
            ax.set_visible(False)
            continue

        # Use common bin range
        all_vals = pd.concat([a_vals, n_vals])
        vmin, vmax = all_vals.quantile(0.01), all_vals.quantile(0.99)
        if vmin == vmax:
            ax.set_visible(False)
            continue

        bins = np.linspace(vmin, vmax, 40)
        a_na, n_na = auas[col].isna().sum(), nfi[col].isna().sum()
        ax.hist(a_vals.clip(vmin, vmax), bins=bins, alpha=0.5, density=True, label=f"AUAS (n={len(a_vals)}, NA={a_na})", color="steelblue")
        ax.hist(n_vals.clip(vmin, vmax), bins=bins, alpha=0.5, density=True, label=f"NFI (n={len(n_vals)}, NA={n_na})", color="tomato")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Feature distributions — activity: '{activity}'", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("figs/feature_distributions.png", dpi=300, bbox_inches="tight")
    print("Saved feature distribution plot to 'figs/feature_distributions.png'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", default="walking")
    args = parser.parse_args()

    auas = load_activity(AUAS_PATH, args.activity)
    nfi = load_activity(NFI_PATH, args.activity)

    print(f"Activity: {args.activity!r}  |  AUAS rows: {len(auas)}  |  NFI rows: {len(nfi)}")

    if auas.empty and nfi.empty:
        print("No data found for this activity in either dataset.")
        return

    feature_cols = get_numeric_feature_cols(auas, nfi)
    print(f"Plotting {len(feature_cols)} shared numeric features: {feature_cols}")
    plot_distributions(auas, nfi, feature_cols, args.activity)


if __name__ == "__main__":
    main()
