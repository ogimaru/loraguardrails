#!/usr/bin/env python
"""
Compare results across all dataset variations.
"""
import os
import yaml
import sys
import pandas as pd
import re
import glob
from pathlib import Path


def extract_metrics_from_reports():
    """Extract metrics from report files for all variations."""
    metrics = []
    
    # Find report files for all variations
    all_report_files = []
    for variation in ["vary_all", "fixed_character", "fixed_history"]:
        # Look for files with variation in the name
        pattern = os.path.join("behavior_reports", "**", f"*{variation}*scoring.txt")
        variation_files = glob.glob(pattern, recursive=True)
        if variation_files:
            print(f"Found {len(variation_files)} reports for variation '{variation}':")
            for f in variation_files:
                print(f"  - {f} (modified: {pd.Timestamp.fromtimestamp(os.path.getmtime(f))})")
                all_report_files.append((f, variation))
    
    if not all_report_files:
        print("No variation-specific report files found. Checking for any report files...")
        # If no variation-specific files found, look for any scoring files
        all_scoring_files = glob.glob(os.path.join("behavior_reports", "**", "*scoring.txt"), recursive=True)
        if all_scoring_files:
            print(f"Found {len(all_scoring_files)} generic report files:")
            for f in all_scoring_files:
                print(f"  - {f} (modified: {pd.Timestamp.fromtimestamp(os.path.getmtime(f))})")
                # Try to determine variation from filename
                for variation in ["vary_all", "fixed_character", "fixed_history"]:
                    if variation in f:
                        all_report_files.append((f, variation))
                        break
                else:
                    # If variation not found in filename, will try to determine from content
                    all_report_files.append((f, None))
        else:
            print("No report files found at all.")
            if os.path.exists("behavior_reports"):
                direct_reports = os.listdir("behavior_reports")
                print(f"Files in behavior_reports: {direct_reports}")
                # Also check subdirectories 
                for subdir in direct_reports:
                    if os.path.isdir(os.path.join("behavior_reports", subdir)):
                        print(f"Files in behavior_reports/{subdir}: {os.listdir(os.path.join('behavior_reports', subdir))}")
            
            return pd.DataFrame(columns=['behavior', 'variation', 'mode', 'overall_accuracy', 'behavior_accuracy', 'natural_accuracy', 'guard_tags'])
    
    # Process each report file
    for report_file, file_variation in all_report_files:
        try:
            filename = os.path.basename(report_file)
            file_time = os.path.getmtime(report_file)
            file_time_str = pd.Timestamp.fromtimestamp(file_time)
            print(f"\nProcessing file: {filename} (modified: {file_time_str})")
            
            # Use the variation found in the filename if available
            variation = file_variation
            
            # Extract mode (single/merged) from filepath or filename
            mode = "unknown"
            if "/single/" in report_file or "_single_" in filename:
                mode = "single"
            elif "/linear_merged/" in report_file or "_linear_merged_" in filename:
                mode = "merged_linear"
            elif "/merged/" in report_file or "_merged_" in filename:
                mode = "merged_svd"
            
            # Extract behavior from filename
            behavior = "unknown"
            
            # First try to extract from path for better accuracy
            if "expert_opinion" in report_file:
                behavior = "expert_opinion"
            elif "politics" in report_file:
                behavior = "politics"
            elif "meeting" in report_file:
                behavior = "meeting"
            else:
                # Fall back to previous method
                filename_parts = filename.split('_')
                if len(filename_parts) > 1:
                    behavior_candidate = filename_parts[1]
                    if behavior_candidate in ["politics", "expert_opinion", "meeting"]:
                        behavior = behavior_candidate
            
            # If variation is not found from the filename, try to determine from content
            if variation is None:
                with open(report_file, 'r') as f:
                    content = f.read()
                    for var in ["vary_all", "fixed_character", "fixed_history"]:
                        if var in content:
                            variation = var
                            print(f"  - Determined variation from content: {variation}")
                            break
                    else:
                        variation = "unknown"
                        print(f"  - Could not determine variation from content")
            
            # Extract metrics from file
            with open(report_file, 'r') as f:
                content = f.read()
                
                overall_acc_match = re.search(r'Overall accuracy with LoRA:\s+(\d+\.\d+)%', content)
                behavior_acc_match = re.search(r'Behavior adherence with LoRA:\s+(\d+\.\d+)%', content)
                natural_acc_match = re.search(r'Natural accuracy with LoRA:\s+(\d+\.\d+)%', content)
                guard_tags_match = re.search(r'Guard tag detection rate:\s+(\d+\.\d+)%', content)
                
                if overall_acc_match:
                    metrics_info = {
                        'behavior': behavior,
                        'variation': variation,
                        'mode': mode,
                        'overall_accuracy': float(overall_acc_match.group(1)) if overall_acc_match else 0,
                        'behavior_accuracy': float(behavior_acc_match.group(1)) if behavior_acc_match else 0,
                        'natural_accuracy': float(natural_acc_match.group(1)) if natural_acc_match else 0,
                        'guard_tags': float(guard_tags_match.group(1)) if guard_tags_match else 0
                    }
                    
                    print(f"  - Extracted metrics: {metrics_info}")
                    metrics.append(metrics_info)
                else:
                    print(f"  - No metrics found in this file")
                    
        except Exception as e:
            print(f"Error processing file {report_file}: {str(e)}")
            continue
    
    result_df = pd.DataFrame(metrics)
    
    return result_df


def compare_and_rank_results():
    """Compare results across variations and output ranking."""
    metrics_df = extract_metrics_from_reports()
    
    if metrics_df.empty:
        print("No evaluation results found. Make sure evaluations ran successfully.")
        return
    
    # Display raw metrics
    print("\nExtracted metrics summary:")
    print(metrics_df[['behavior', 'variation', 'mode', 'overall_accuracy', 'behavior_accuracy', 'natural_accuracy', 'guard_tags']].to_string(index=False))
    
    # Calculate composite score (simple average of normalized ranks)
    # Handle division by zero by replacing 0 with 1 for normalization
    max_overall = metrics_df['overall_accuracy'].max() if metrics_df['overall_accuracy'].max() > 0 else 1
    max_behavior = metrics_df['behavior_accuracy'].max() if metrics_df['behavior_accuracy'].max() > 0 else 1
    max_natural = metrics_df['natural_accuracy'].max() if metrics_df['natural_accuracy'].max() > 0 else 1
    max_guard = metrics_df['guard_tags'].max() if metrics_df['guard_tags'].max() > 0 else 1
    
    # Normalize scores
    metrics_df['overall_norm'] = metrics_df['overall_accuracy'] / max_overall
    metrics_df['behavior_norm'] = metrics_df['behavior_accuracy'] / max_behavior
    metrics_df['natural_norm'] = metrics_df['natural_accuracy'] / max_natural
    metrics_df['guard_norm'] = metrics_df['guard_tags'] / max_guard
    
    # Calculate composite score
    metrics_df['composite_score'] = (
        metrics_df['overall_norm'] + 
        metrics_df['behavior_norm'] + 
        metrics_df['natural_norm'] + 
        metrics_df['guard_norm']
    ) / 4
    
    # Sort by composite score
    metrics_df = metrics_df.sort_values('composite_score', ascending=False)
    
    # Display ranked results
    print("\nRanked results by composite score:")
    print(metrics_df[['behavior', 'variation', 'mode', 'composite_score', 'overall_accuracy', 'behavior_accuracy', 'natural_accuracy', 'guard_tags']].to_string(index=False))
    
    # Save results to CSV
    output_file = "variation_comparison_results.csv"
    metrics_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    compare_and_rank_results() 