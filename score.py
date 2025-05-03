#!/usr/bin/env python
"""
Scores the predictions against the ground truth.

Calculates accuracy for both 'task' and 'status' predictions.

Usage: python score.py data/test.csv preds.csv
"""

import argparse
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def main(ground_truth_path, predictions_path):
    try:
        # Read ground truth CSV assuming no header, provide column names
        df_true = pd.read_csv(ground_truth_path, header=None, names=['note_id', 'text', 'task', 'completed'])
        print(f"Successfully read ground truth file: {ground_truth_path}")
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading ground truth file {ground_truth_path}: {e}")
        sys.exit(1)

    try:
        df_pred = pd.read_csv(predictions_path)
        print(f"Successfully read predictions file: {predictions_path}")
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading predictions file {predictions_path}: {e}")
        sys.exit(1)

    # --- Input Validation ---
    required_true_cols = ['task', 'completed']
    required_pred_cols = ['task', 'completed']

    missing_true = [col for col in required_true_cols if col not in df_true.columns]
    if missing_true:
        print(f"Error: Ground truth file ({ground_truth_path}) is missing required columns: {missing_true}")
        print(f"Found columns: {df_true.columns.tolist()}")
        sys.exit(1)

    missing_pred = [col for col in required_pred_cols if col not in df_pred.columns]
    if missing_pred:
        print(f"Error: Predictions file ({predictions_path}) is missing required columns: {missing_pred}")
        print(f"Found columns: {df_pred.columns.tolist()}")
        sys.exit(1)

    if len(df_true) != len(df_pred):
        print(f"Warning: Mismatched row counts between files! ({len(df_true)} vs {len(df_pred)}). Scoring based on available rows.")
        # Optional: Add logic here to align rows based on an ID if necessary

    # --- Scoring ---
    print("\n--- Scoring Results ---")

    # Score Task Prediction
    try:
        task_accuracy = accuracy_score(df_true['task'], df_pred['task'])
        print(f"\nTask Prediction Accuracy: {task_accuracy:.4f}")
        print("Task Classification Report:")
        print(classification_report(df_true['task'], df_pred['task'], zero_division=0))
    except Exception as e:
        print(f"Error calculating task score: {e}")


    # Score Status Prediction
    # Map predicted '2' (not_sure) to '0' (not_completed) for fair comparison with 0/1 ground truth
    try:
        # Create copies to avoid modifying original dataframes
        true_status = df_true['completed'].copy()
        pred_status = df_pred['completed'].copy()

        # Map predicted 2s to 0s
        pred_status[pred_status == 2] = 0

        status_accuracy = accuracy_score(true_status, pred_status)
        print(f"\nStatus Prediction Accuracy (mapping predicted 'not_sure' [2] to 'not_completed' [0]): {status_accuracy:.4f}")
        print("Status Classification Report (after mapping 2->0):")
        # Use the modified series for the report
        print(classification_report(true_status, pred_status, zero_division=0))
    except Exception as e:
        print(f"Error calculating status score: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score model predictions against ground truth.")
    parser.add_argument("ground_truth_csv", help="Path to the ground truth CSV file (e.g., data/test.csv)")
    parser.add_argument("predictions_csv", help="Path to the predictions CSV file (e.g., preds.csv)")

    if len(sys.argv) != 3:
         # Fallback for direct script execution without argparse
         if len(sys.argv) == 3 and sys.argv[1].lower().endswith('.csv') and sys.argv[2].lower().endswith('.csv'):
             print("Running with direct arguments...")
             main(sys.argv[1], sys.argv[2])
         else:
             parser.print_help()
             print("\nExample Usage: python score.py data/test.csv preds.csv")
             sys.exit(1)
    else:
         args = parser.parse_args()
         main(args.ground_truth_csv, args.predictions_csv)