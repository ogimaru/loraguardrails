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

            # Extract metrics from file
            with open(report_file, 'r') as f:
                content = f.read()
                
                # Extract "with LoRA" metrics
                overall_acc_match = re.search(r'Overall accuracy with LoRA:\s+(\d+\.\d+)%', content)
                behavior_acc_match = re.search(r'Behavior adherence with LoRA:\s+(\d+\.\d+)%', content)
                natural_acc_match = re.search(r'Natural accuracy with LoRA:\s+(\d+\.\d+)%', content)
                guard_tags_match = re.search(r'Guard tag detection rate:\s+(\d+\.\d+)%', content)
                
                # Extract "without LoRA" metrics (baseline performance)
                behavior_acc_without_lora_match = re.search(r'Behavior adherence without LoRA:\s+(\d+\.\d+)%', content)
                natural_acc_without_lora_match = re.search(r'Natural accuracy without LoRA:\s+(\d+\.\d+)%', content)
                
                # Use the first found set of metrics as the primary check for validity
                if overall_acc_match or behavior_acc_match or natural_acc_match or guard_tags_match:
                    metrics_info = {
                        'behavior': behavior,
                        'variation': variation,
                        'mode': mode,
                        'overall_accuracy': float(overall_acc_match.group(1)) if overall_acc_match else 0.0,
                        'behavior_accuracy': float(behavior_acc_match.group(1)) if behavior_acc_match else 0.0,
                        'natural_accuracy': float(natural_acc_match.group(1)) if natural_acc_match else 0.0,
                        'guard_tags': float(guard_tags_match.group(1)) if guard_tags_match else 0.0, 
                        'behavior_accuracy_without_lora': float(behavior_acc_without_lora_match.group(1)) if behavior_acc_without_lora_match else 0.0,
                        'natural_accuracy_without_lora': float(natural_acc_without_lora_match.group(1)) if natural_acc_without_lora_match else 0.0,
                    }
                    
                    print(f"  - Extracted metrics: {metrics_info}")
                    metrics.append(metrics_info)
                else:
                    # Check if baseline metrics were found even if 'with LoRA' were not
                    if behavior_acc_without_lora_match or natural_acc_without_lora_match:
                         metrics_info = {
                            'behavior': behavior,
                            'variation': variation,
                            'mode': mode,
                            'overall_accuracy': 0.0,
                            'behavior_accuracy': 0.0,
                            'natural_accuracy': 0.0,
                            'guard_tags': 0.0,
                            'behavior_accuracy_without_lora': float(behavior_acc_without_lora_match.group(1)) if behavior_acc_without_lora_match else 0.0,
                            'natural_accuracy_without_lora': float(natural_acc_without_lora_match.group(1)) if natural_acc_without_lora_match else 0.0,
                         }
                         print(f"  - Extracted only baseline metrics: {metrics_info}")
                         metrics.append(metrics_info)
                    else:
                        print(f"  - No relevant metrics found in this file")
                    
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
    # Expand columns displayed slightly to include baseline behavior accuracy
    display_cols = ['behavior', 'variation', 'mode', 'overall_accuracy', 'behavior_accuracy', 'natural_accuracy', 'guard_tags', 'behavior_accuracy_without_lora']
    # Only display columns that actually exist in the DataFrame
    display_cols = [col for col in display_cols if col in metrics_df.columns]
    print(metrics_df[display_cols].to_string(index=False))
    

    
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
    


    print("\n\n--- Manuscript Table 1: Behavior Adherence Comparison (vary_all variation) ---")
    try:
        # Filter for the specific variation (e.g., 'vary_all')
        df_vary_all = metrics_df[metrics_df['variation'] == 'vary_all'].copy()
        
        behaviors_order = ['meeting', 'politics', 'expert_opinion'] 
        behavior_display_map = {'meeting': 'Meeting', 'politics': 'Politics', 'expert_opinion': 'Expert-Opinion'}
        
        table1_data = {}
        
        # Get Baseline (No Adapter) - Should be consistent across modes for the same behavior
        baseline_data = df_vary_all[df_vary_all['mode'] == 'single'].set_index('behavior')['behavior_accuracy_without_lora'].to_dict()
        table1_data['No Adapter'] = {behavior_display_map.get(b, b): baseline_data.get(b, pd.NA) for b in behaviors_order}

        # Get Single Adapter 
        single_data = df_vary_all[df_vary_all['mode'] == 'single'].set_index('behavior')['behavior_accuracy'].to_dict()
        table1_data['Single Adapter'] = {behavior_display_map.get(b, b): single_data.get(b, pd.NA) for b in behaviors_order}

        # Get Merged Adapter (SVD)
        merged_data = df_vary_all[df_vary_all['mode'] == 'merged_svd'].set_index('behavior')['behavior_accuracy'].to_dict()
        table1_data['Merged Adapter'] = {behavior_display_map.get(b, b): merged_data.get(b, pd.NA) for b in behaviors_order}

        # Create DataFrame for formatting
        table1_df = pd.DataFrame(table1_data).T # Transpose to get modes as rows
        # Reorder columns according to desired display
        table1_df = table1_df[[behavior_display_map[b] for b in behaviors_order if behavior_display_map[b] in table1_df.columns]]
        
        if not table1_df.empty:
            print(table1_df.to_string(float_format="%.1f", na_rep="N/A"))
        else:
            print("Could not generate Table 1: No data found for 'vary_all' variation with required modes (single, merged_svd).")
            
    except Exception as e:
        print(f"Could not generate Table 1 due to error: {e}") 

    print("\n\n--- Manuscript Table 2: Detailed Single Adapter Performance (vary_all variation) ---")
    try:
        # Filter for single adapter mode and vary_all variation
        df_single_vary_all = metrics_df[(metrics_df['mode'] == 'single') & (metrics_df['variation'] == 'vary_all')].copy()
        
        if not df_single_vary_all.empty:
            # Select and rename columns according to Table 2
            table2_df = df_single_vary_all[[
                'behavior', 
                'behavior_accuracy',              
                'natural_accuracy',              
                'behavior_accuracy_without_lora', 
                'guard_tags'                     
            ]].copy()
            
            table2_df.rename(columns={
                'behavior': 'Guardrail',
                'behavior_accuracy': 'LoRA',
                'natural_accuracy': 'Neutral',
                'behavior_accuracy_without_lora': 'Base',
                'guard_tags': 'Tags'
            }, inplace=True)

            # Use nicer behavior names
            behavior_display_map_t2 = {'meeting': 'Meeting', 'politics': 'Politics', 'expert_opinion': 'Expert Op.'}
            table2_df['Guardrail'] = table2_df['Guardrail'].map(behavior_display_map_t2).fillna(table2_df['Guardrail'])
 
            table2_df = table2_df.set_index('Guardrail')
             
            desired_row_order = ['Politics', 'Meeting', 'Expert Op.']
            table2_df = table2_df.reindex([idx for idx in desired_row_order if idx in table2_df.index])

            print(table2_df.to_string(float_format="%.1f"))
        else:
            print("Could not generate Table 2: No data found for 'single' mode and 'vary_all' variation.")
            
    except Exception as e:
        print(f"Could not generate Table 2 due to error: {e}") 

    # Save results to CSV
    output_file = "variation_comparison_results.csv"
    metrics_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    compare_and_rank_results() 