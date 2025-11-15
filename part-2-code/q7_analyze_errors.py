#!/usr/bin/env python3
"""Analyze errors in model predictions for qualitative analysis"""

from utils import load_queries_and_records
import re

def analyze_errors(gt_sql_path, pred_sql_path, pred_records_path, gt_records_path):
    """Analyze different types of errors in predictions"""
    
    gt_sql, _, _ = load_queries_and_records(gt_sql_path, gt_records_path)
    pred_sql, pred_records, error_msgs = load_queries_and_records(pred_sql_path, pred_records_path)
    _, gt_records, _ = load_queries_and_records(gt_sql_path, gt_records_path)
    
    total = len(pred_sql)
    
    # Error type 1: SQL Syntax Errors
    syntax_errors = sum(1 for msg in error_msgs if msg != "")
    syntax_error_examples = []
    for i, msg in enumerate(error_msgs):
        if msg != "":
            syntax_error_examples.append((i, pred_sql[i][:100] + "...", msg))
            if len(syntax_error_examples) >= 3:
                break
    
    # Error type 2: Incorrect table/column names (syntax correct but execution fails)
    # These are already captured in syntax_errors, but we can separate them
    
    # Error type 3: Logical errors (syntax correct, executes, but wrong results)
    logical_errors = 0
    logical_error_examples = []
    for i, (gt_rec, pred_rec) in enumerate(zip(gt_records, pred_records)):
        if error_msgs[i] == "":  # No syntax error
            if set(gt_rec) != set(pred_rec):  # Results don't match
                logical_errors += 1
                if len(logical_error_examples) < 3:
                    # Find the NL query for context
                    with open('data/dev.nl', 'r') as f:
                        nl_queries = [line.strip() for line in f.readlines()]
                    logical_error_examples.append((
                        nl_queries[i][:80] + "...",
                        pred_sql[i][:100] + "..."
                    ))
    
    print("=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    print(f"\n1. SQL Syntax Errors: {syntax_errors}/{total} ({syntax_errors/total*100:.2f}%)")
    print("   Examples:")
    for idx, sql, err in syntax_error_examples[:3]:
        print(f"   - Query {idx}: {sql}")
        print(f"     Error: {err[:100]}")
    
    print(f"\n2. Logical Errors (Wrong Results): {logical_errors}/{total} ({logical_errors/total*100:.2f}%)")
    print("   Examples:")
    for nl, sql in logical_error_examples[:3]:
        print(f"   - NL: {nl}")
        print(f"     SQL: {sql}")
    
    print(f"\n3. Exact Match (Correct): {total - syntax_errors - logical_errors}/{total}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python analyze_errors.py <gt_sql> <pred_sql> <pred_records> <gt_records>")
        sys.exit(1)
    
    analyze_errors(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
