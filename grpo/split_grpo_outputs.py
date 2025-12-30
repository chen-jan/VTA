#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GRPO Stage 3 full-dataset results into per-stock CSVs for the backbone.")
    parser.add_argument("--results-dir", required=True,
                        help="Results directory where grpo2_full_dataset_results.csv is stored.")
    parser.add_argument("--full-x-path", required=True,
                        help="Path to the dataset full_x CSV (used to fetch the 'stock' column for grouping).")
    parser.add_argument("--model", required=False, default="",
                        help="Hugging Face model name used (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Used for output path labeling.")
    parser.add_argument("--dataset-label", required=False, default="stocknet",
                        help="Dataset label folder name under backbone/data/grpo_input (e.g., 'stocknet').")
    parser.add_argument("--output-root", required=False, default="",
                        help="Optional override for the root output dir. Default: <repo_root>/backbone/data/grpo_input")
    parser.add_argument("--output-columns", nargs="+", default=["grpo_prediction", "prompt_question", "full_answer"],
                        help="Columns to write to each per-stock CSV.")
    return parser.parse_args()


def get_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))


def get_model_label(model_name: str) -> str:
    if not model_name:
        return "model"
    # Prefer the last path component if it contains '/'
    parts = model_name.split("/")
    last = parts[-1] if parts else model_name
    return last.strip()


def main() -> None:
    args = parse_args()

    results_csv = os.path.join(args.results_dir, "grpo2_full_dataset_results.csv")
    if not os.path.exists(results_csv):
        print(f"Error: results CSV not found at {results_csv}")
        sys.exit(1)

    if not os.path.exists(args.full_x_path):
        print(f"Error: full_x CSV not found at {args.full_x_path}")
        sys.exit(1)

    print(f"Reading results from: {results_csv}")
    df = pd.read_csv(results_csv)

    # Try to ensure 'example_index' is available for alignment if needed
    has_example_index_col = "example_index" in df.columns

    print(f"Reading full_x from: {args.full_x_path}")
    full_x_df = pd.read_csv(args.full_x_path)
    if "stock" not in full_x_df.columns:
        print("Error: 'stock' column not found in full_x CSV. Cannot split per stock.")
        sys.exit(1)

    # Align 'stock' to df
    # Primary: if df already has 'stock', use it.
    if "stock" in df.columns:
        stock_series = df["stock"].copy()
    else:
        # Fallback: align by row order; if 'example_index' exists and is unique, try aligning by that
        if has_example_index_col and df["example_index"].is_monotonic_increasing:
            # If example_index looks like row ids, try aligning by index values
            try:
                stock_series = full_x_df.loc[df["example_index"], "stock"].reset_index(drop=True)
            except Exception:
                # fallback to direct positional alignment
                stock_series = full_x_df["stock"].iloc[:len(df)].reset_index(drop=True)
        else:
            stock_series = full_x_df["stock"].iloc[:len(df)].reset_index(drop=True)

        if len(stock_series) != len(df):
            print("Warning: Length mismatch between results and full_x. Truncating to the minimum length.")
            min_len = min(len(stock_series), len(df))
            df = df.iloc[:min_len].copy()
            stock_series = stock_series.iloc[:min_len].copy()

        df["stock"] = stock_series

    # Build output directory
    repo_root = get_repo_root()
    output_root = args.output_root or os.path.join(repo_root, "backbone", "data", "grpo_input")
    model_label = get_model_label(args.model)
    output_dir = os.path.join(output_root, args.dataset_label, model_label)
    os.makedirs(output_dir, exist_ok=True)

    cols = args.output_columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Error: Required columns missing in results CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Writing per-stock CSVs to: {output_dir}")
    num_files = 0
    for stock_name, stock_df in df.groupby("stock"):
        stock_df = stock_df.reset_index(drop=True)
        out_path = os.path.join(output_dir, f"{stock_name}.csv")
        stock_df.to_csv(
            out_path,
            columns=cols,
            index=True,
            index_label="example_index",
        )
        num_files += 1
        print(f"{out_path}: {len(stock_df)} rows")

    print(f"Done. Wrote {num_files} files to {output_dir}")


if __name__ == "__main__":
    main()


