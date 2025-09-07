from pathlib import Path
import sys

import pandas as pd


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Define output directory
FINAL_OUTPUT_DIR = PROJECT_ROOT / "output"
FINAL_OUTPUT_DIR.mkdir(exist_ok=True)

# Define paths to individual experiment summary files
SUMMARY_PATHS = {
    "robustness": PROJECT_ROOT / "output/robustness/summary.csv",
    "tta": PROJECT_ROOT / "output/tta/summary.csv",
    "ensembling": PROJECT_ROOT / "output/ensembling/summary.csv",
    "tiling": PROJECT_ROOT / "output/tiling/summary_tiling.csv",
}


def load_and_process_robustness(path):
    """Loads and processes the robustness summary, focusing on the baseline."""
    if not path.exists():
        print(f"Warning: {path} not found. Skipping.")
        return None
    df = pd.read_csv(path)
    # Filter for baseline results (no corruption)
    baseline_df = df[df["corruption"] == "baseline"].copy()
    baseline_df["experiment"] = "Baseline"
    return baseline_df[["experiment", "model", "mAP", "AP_S"]]


def load_and_process_tta(path):
    """Loads and processes the TTA summary."""
    if not path.exists():
        print(f"Warning: {path} not found. Skipping.")
        return None
    df = pd.read_csv(path)
    df["experiment"] = "TTA"
    # The 'Baseline' model here is just the raw model, let's rename for clarity
    df.loc[df["method"] == "Baseline", "experiment"] = "Baseline"
    df = df.rename(columns={"method": "model_desc"})
    return df[["experiment", "model", "model_desc", "mAP", "AP_S"]]


def load_and_process_ensembling(path):
    """Loads and processes the ensembling summary."""
    if not path.exists():
        print(f"Warning: {path} not found. Skipping.")
        return None
    df = pd.read_csv(path)
    df["experiment"] = "Ensembling"
    df.loc[df["model"] != "WBF Ensemble", "experiment"] = "Baseline"
    return df[["experiment", "model", "mAP", "AP_S"]]


def load_and_process_tiling(path):
    """Loads and processes the tiling summary."""
    if not path.exists():
        print(f"Warning: {path} not found. Skipping.")
        return None
    df = pd.read_csv(path)
    df["experiment"] = "Tiling"
    return df[["experiment", "model", "mAP", "AP_S"]]


def main():
    """
    Main function to load all experiment summaries, consolidate them,
    and save a final summary table.
    """
    print("--- Consolidating All Experiment Results ---")

    processed_dfs = []

    # --- Load and process each summary file individually ---

    # Robustness (Baseline)
    if SUMMARY_PATHS["robustness"].exists():
        df = pd.read_csv(SUMMARY_PATHS["robustness"])
        df = df[df["corruption"] == "baseline"].copy()
        df["experiment"] = "Baseline"
        processed_dfs.append(df[["experiment", "model", "mAP", "AP_S"]])
        print(f"Loaded {len(df)} baseline results from Robustness summary.")

    # TTA
    if SUMMARY_PATHS["tta"].exists():
        df = pd.read_csv(SUMMARY_PATHS["tta"])
        # Separate baseline from TTA results
        baseline = df[df["method"] == "Baseline"].copy()
        baseline["experiment"] = "Baseline"
        tta = df[df["method"] == "TTA"].copy()
        tta["experiment"] = "TTA"
        processed_dfs.append(baseline[["experiment", "model", "mAP", "AP_S"]])
        processed_dfs.append(tta[["experiment", "model", "mAP", "AP_S"]])
        print(f"Loaded {len(df)} results from TTA summary.")

    # Ensembling
    if SUMMARY_PATHS["ensembling"].exists():
        df = pd.read_csv(SUMMARY_PATHS["ensembling"])
        # Separate baseline from WBF results
        wbf = df[df["model"] == "WBF Ensemble"].copy()
        wbf["experiment"] = "Ensembling (WBF)"
        processed_dfs.append(wbf[["experiment", "model", "mAP", "AP_S"]])
        print(f"Loaded {len(wbf)} results from Ensembling summary.")

    # Tiling
    if SUMMARY_PATHS["tiling"].exists():
        df = pd.read_csv(SUMMARY_PATHS["tiling"])
        df["experiment"] = "Tiling"
        processed_dfs.append(df[["experiment", "model", "mAP", "AP_S"]])
        print(f"Loaded {len(df)} results from Tiling summary.")

    if not processed_dfs:
        print("No result summaries found to process. Exiting.")
        return

    # Combine all dataframes
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Clean up model names
    final_df["model"] = final_df["model"].str.replace(".pt", "", regex=False)
    final_df["model"] = final_df["model"].str.replace("_baseline", "", regex=False)
    final_df["model"] = final_df["model"].str.replace("_aug_20250603", "", regex=False)

    # Standardize model names for grouping
    final_df.loc[final_df["model"] == "WBF Ensemble", "model"] = "Ensemble"

    # Drop duplicates to keep one 'Baseline' entry per model
    final_df = final_df.drop_duplicates(subset=["experiment", "model"])

    # Pivot table for comparison
    pivot_df = final_df.pivot_table(index="model", columns="experiment", values="AP_S").reset_index()

    # Sort for better presentation
    pivot_df = pivot_df.sort_values(by="Baseline", ascending=False)

    # Save the final summary
    final_summary_path = FINAL_OUTPUT_DIR / "final_summary.csv"
    pivot_df.to_csv(final_summary_path, index=False, float_format="%.4f")

    print("\n--- Final Consolidated Summary ---")
    print(pivot_df.to_string())
    print(f"\nFinal summary saved to: {final_summary_path}")


if __name__ == "__main__":
    main()
