#!/usr/bin/env python3
"""Get all metrics (SQL EM, Record EM, F1) for Q6 table"""

from utils import compute_metrics

# Dev set metrics
sql_em, record_em, record_f1, error_msgs = compute_metrics(
    'data/dev.sql',
    'results/t5_ft_ft_experiment_dev.sql',
    'records/ground_truth_dev.pkl',
    'records/t5_ft_ft_experiment_dev.pkl'
)

print("=" * 60)
print("DEV SET METRICS")
print("=" * 60)
print(f"SQL Exact Match (Query EM): {sql_em:.4f} ({sql_em*100:.2f}%)")
print(f"Record Exact Match: {record_em:.4f} ({record_em*100:.2f}%)")
print(f"Record F1: {record_f1:.4f} ({record_f1*100:.2f}%)")
print("=" * 60)
print("\nFor Table 4:")
print(f"Query EM: {sql_em*100:.2f}")
print(f"F1 score: {record_f1*100:.2f}")

