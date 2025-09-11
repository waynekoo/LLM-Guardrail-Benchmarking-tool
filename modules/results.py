# Handles result storage, export, and comparison
import os
import json
import pandas as pd

def load_result(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_metrics(logs):
    # Calculate confusion matrix and metrics
    cm = [[0, 0], [0, 0]]
    for log in logs:
        actual = 0 if log["label"] == "malicious" else 1
        pred = 0 if log["evaluation"]["isSafeResult"] == False else 1
        cm[actual][pred] += 1
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total
    }

def diff_logs(logs1, logs2):
    # Compare logs by input (assumes 'input' field exists)
    df1 = pd.DataFrame(logs1)
    df2 = pd.DataFrame(logs2)
    if "input" in df1.columns and "input" in df2.columns:
        merged = pd.merge(df1, df2, on="input", suffixes=("_1", "_2"))
        diff = merged[merged["evaluation.isSafeResult_1"] != merged["evaluation.isSafeResult_2"]]
        return diff
    return pd.DataFrame()

def get_dataset_info(result_data):
    logs = result_data.get("results", [])
    # Defensive: some logs may have dataset=None or missing
    datasets = [log.get("dataset") for log in logs if log.get("dataset")]
    unique_datasets = sorted(set(datasets))
    row_counts = {ds: datasets.count(ds) for ds in unique_datasets}
    return {"datasets": unique_datasets, "row_counts": row_counts}
