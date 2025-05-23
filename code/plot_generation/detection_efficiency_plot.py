import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import glob
from typing import List
import re

def print_available_metrics_new_format(adapter_mode: str = "single"):
    """Explain where the script now looks for metrics files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path is from code/plot_generation/ to code/guardrail_evaluation/behavior_reports/
    base_report_dir = os.path.join(script_dir, "..", "guardrail_evaluation", "behavior_reports")
    report_mode_dir = os.path.join(base_report_dir, adapter_mode)

    print("\nThis script now attempts to load metrics from '*.txt' scoring report files.")
    print(f"It looks for behavior-specific scoring reports (e.g., '*_scoring.txt') in:")
    print(f"  {report_mode_dir}")
    print(f"And the Llama Guard report for politics from:")
    print(f"  {os.path.join(base_report_dir, 'politics_guardrail_report_llama_guard.txt')}")

    print("\nDiscovered scoring reports in the target directory:")
    scoring_files = glob.glob(os.path.join(report_mode_dir, "*_scoring.txt"))
    if scoring_files:
        for i, file_path in enumerate(scoring_files):
            print(f"  {i+1}. {os.path.basename(file_path)}")
    else:
        print(f"  No '*_scoring.txt' reports found in {report_mode_dir}")
    
    llama_guard_specific_report = os.path.join(base_report_dir, 'politics_guardrail_report_llama_guard.txt')
    if os.path.exists(llama_guard_specific_report):
        print(f"  - {os.path.basename(llama_guard_specific_report)} (for Llama Guard comparison)")
    else:
        print(f"  - Llama Guard report not found at {llama_guard_specific_report}")

def load_metrics_from_scoring_reports(adapter_mode: str = "single", requested_behaviors: List[str] = None):
    """Load metrics for specified behaviors from _scoring.txt files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_report_dir = os.path.join(script_dir, "..", "guardrail_evaluation", "behavior_reports")
    
    metrics_by_behavior = {} # This will store the final data for plotting

    # Determine the specific directory for the given adapter_mode
    report_mode_dir = os.path.join(base_report_dir, adapter_mode)
    if not os.path.exists(report_mode_dir):
        print(f"Warning: Report directory not found at {report_mode_dir}")
        # Don't return yet, still need to check for Llama Guard file if requested/relevant

    print(f"Searching for scoring reports in: {report_mode_dir}")
    # Example filenames: report_politics_single_vary_all_scoring.txt, report_expert_opinion_merged_vary_all_scoring.txt
    scoring_files = glob.glob(os.path.join(report_mode_dir, "*_scoring.txt"))

    # Regex patterns for metrics from _scoring.txt files (generated by testing_lora_adapters.py) 
    patterns = {
        "behavior_detection_with_lora": r"Behavior adherence with LoRA:\s*([\d.]+?)%",
        "behavior_detection_without_lora": r"Behavior adherence without LoRA:\s*([\d.]+?)%",
        "guard_tag_usage": r"Guard tag detection rate:\s*([\d.]+?)%",
        "neutral_detection_with_lora": r"Natural accuracy with LoRA:\s*([\d.]+?)%",
        "neutral_detection_without_lora": r"Natural accuracy without LoRA:\s*([\d.]+?)%"
    }

    parsed_behaviors_from_reports = {}

    for report_file_path in scoring_files:
        filename = os.path.basename(report_file_path) 
        
        match_behavior = re.match(r"report_([a-zA-Z0-9_]+?)_(?:single|merged|linear_merged)(?:_[a-zA-Z0-9_]+?)?_scoring\.txt", filename)
        if not match_behavior: 
            match_behavior = re.match(r"report_([a-zA-Z0-9_]+?)_(?:single|merged|linear_merged)_scoring\.txt", filename)
        
        if not match_behavior:
            print(f"Warning: Could not parse behavior name from filename: {filename}")
            continue
            
        behavior_name = match_behavior.group(1)

        if requested_behaviors and behavior_name not in requested_behaviors:
            print(f"Skipping {behavior_name} as it's not in requested_behaviors: {requested_behaviors}")
            continue
        
        try:
            with open(report_file_path, 'r') as f:
                content = f.read()
            
            current_metrics = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    current_metrics[key] = float(match.group(1)) / 100.0  
                else:
                    print(f"Warning: Metric for '{key}' not found in {filename}. Defaulting to 0.0.")
                    current_metrics[key] = 0.0 
            
            if any(current_metrics.values()): 
                parsed_behaviors_from_reports[behavior_name] = current_metrics
                print(f"Loaded metrics for {behavior_name} from {report_file_path}")
            else:
                print(f"Warning: No metrics successfully parsed from {filename} for behavior {behavior_name}.")

        except Exception as e:
            print(f"Error processing file {report_file_path}: {str(e)}")
            
    # Handle Llama Guard report separately 
    llama_guard_report_path = os.path.join(base_report_dir, "politics_guardrail_report_llama_guard.txt")
    if os.path.exists(llama_guard_report_path):
        # Add Llama Guard metrics if "politics" is a requested behavior or if no specific behaviors are requested (implying plot all) 
        should_load_llama_guard = True # Load by default if file exists
        if requested_behaviors and "politics" not in requested_behaviors and "politics_llama_guard" not in requested_behaviors:
            should_load_llama_guard = False

        if should_load_llama_guard:
            try:
                with open(llama_guard_report_path, 'r') as f:
                    content = f.read()
                
                llama_metric_val = 0.0 
                match_behavioral = re.search(r"Behavioral prompt accuracy:\s*([\d.]+?)%", content)
                match_overall = re.search(r"Overall Accuracy:\s*([\d.]+?)%", content)

                if match_behavioral:
                    llama_metric_val = float(match_behavioral.group(1)) / 100.0
                elif match_overall:
                    llama_metric_val = float(match_overall.group(1)) / 100.0
                else:
                    print(f"Warning: Llama Guard primary accuracy metric not found in {llama_guard_report_path}")

                # Reusing metrics by behavior, adding others as placeholder values.
                metrics_by_behavior["politics_llama_guard"] = {
                    "behavior_detection_with_lora": llama_metric_val,
                    "behavior_detection_without_lora": 0.0, 
                    "guard_tag_usage": 0.0, 
                    "neutral_detection_with_lora": 0.0, 
                    "neutral_detection_without_lora": 0.0 
                }
                print(f"Loaded metrics for politics_llama_guard from {llama_guard_report_path}")

            except Exception as e:
                print(f"Error processing Llama Guard report {llama_guard_report_path}: {str(e)}")

    # Populate the final metrics_by_behavior dictionary for plotting
    # This ensures that if specific behaviors were requested, only those are included.
    if requested_behaviors:
        for behavior_key in requested_behaviors:
            if behavior_key in parsed_behaviors_from_reports:
                metrics_by_behavior[behavior_key] = parsed_behaviors_from_reports[behavior_key]
             
    else: 
        for behavior_key, metrics_data in parsed_behaviors_from_reports.items():
            metrics_by_behavior[behavior_key] = metrics_data
            
    if not metrics_by_behavior:
         print(f"Warning: No metrics ultimately loaded for adapter mode '{adapter_mode}' and requested behaviors: {requested_behaviors}. Plot may be empty.")

    return metrics_by_behavior

def plot_detection_efficiency(metrics_by_behavior, output_file="plots/detection_efficiency.png"):
    """Create plots for detection efficiency using Seaborn."""
    
    plt.rcParams.update({
        'figure.figsize': (12, 6),         
        'font.family': 'serif',
        'font.size': 24,                   
        'axes.titlesize': 26,              
        'axes.labelsize': 24,              
        'xtick.labelsize': 20,             
        'ytick.labelsize': 20,             
        'legend.fontsize': 20,             
        'lines.linewidth': 2,              
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
     
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
     
    palette = {
        'Baseline': 'royalblue',
        'With Guardrail': 'lightgreen',
        'LLama Guard': 'orange'
    }
    
    # First plot: Sets of two bars for each behavior type
    behaviors = [b for b in metrics_by_behavior.keys() if b != "politics_llama_guard"]
     
    data = []
    for behavior in behaviors:
        # Get behavior detection rates (baseline = without_lora, guardrail = with_lora)
        baseline = metrics_by_behavior[behavior].get('behavior_detection_without_lora', 0) * 100
        guardrail = metrics_by_behavior[behavior].get('behavior_detection_with_lora', 0) * 100
        
        # Replace underscore with space for plot labeling
        plot_label = behavior.replace("_", " ").capitalize()

        data.append({
            'Behavior': plot_label, # Use modified label
            'Detection Rate': baseline,
            'Type': 'Baseline'
        })
        data.append({
            'Behavior': plot_label, # Use modified label
            'Detection Rate': guardrail,
            'Type': 'With Guardrail'
        })
    
    df = pd.DataFrame(data)
     
    sns.barplot(x='Behavior', y='Detection Rate', hue='Type', data=df, palette=palette, ax=ax1, width=0.8)
    
    ax1.set_ylabel('Behavior Detection Rate (%)')
    ax1.set_title('')
     
    ax1.get_legend().remove()
     
    ax1.set_xlabel('')
     
    for p in ax1.patches:
        if p.get_height() > 1.0: 
            ax1.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=20)
    
    # Second plot: Politics guardrail vs LLama Guard (if politics exists)    
    if 'politics' in metrics_by_behavior and 'politics_llama_guard' in metrics_by_behavior:
        politics_detection = metrics_by_behavior['politics'].get('behavior_detection_with_lora', 0) * 100
        llama_guard_detection = metrics_by_behavior['politics_llama_guard'].get('behavior_detection_with_lora', 0) * 100
        
        comparison_data = [
            {
                'Behavior': 'Comparison', 
                'Detection Rate': politics_detection,
                'Type': 'With Guardrail'
            },
            {
                'Behavior': 'Comparison', 
                'Type': 'LLama Guard',
                'Detection Rate': llama_guard_detection
            }
        ]
        
        # Add the comparison data to the same DataFrame
        extended_df = pd.concat([df, pd.DataFrame(comparison_data)], ignore_index=True)
        
        comparison_df = extended_df[extended_df['Behavior'] == 'Comparison']
        sns.barplot(x='Behavior', y='Detection Rate', hue='Type', 
                   data=comparison_df,
                   palette={'With Guardrail': palette['With Guardrail'], 'LLama Guard': palette['LLama Guard']}, 
                   ax=ax2, width=0.8)
        
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylabel('') 
        ax2.set_title('')
        ax2.set_xlabel('')
         
        ax2.set_xticklabels(['Politics'])
        
        # Remove the legend from the right plot
        ax2.get_legend().remove()
         
        for p in ax2.patches:
            if p.get_height() > 1.0:  
                ax2.annotate(f'{p.get_height():.1f}%',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=20)
                            
        # Create a combined legend for both plots and place it in between the figures
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=palette['Baseline'], lw=0, marker='s', markersize=15, label='Baseline'),
            Line2D([0], [0], color=palette['With Guardrail'], lw=0, marker='s', markersize=15, label='With Guardrail'),
            Line2D([0], [0], color=palette['LLama Guard'], lw=0, marker='s', markersize=15, label='LLama Guard')
        ]
        # Use figure-level legend instead of axis-level
        # Position it to the right of the y-axis labels of the left figure
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.35, 0.5), framealpha=0.9, fontsize=20)
    else:
        ax2.set_visible(False)
    
    # Adjust the spacing to accommodate the legend in the middle
    plt.subplots_adjust(wspace=0.1) 
    
    # Ensure the plots directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
     
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    print(f"Plot saved as {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate detection efficiency plots from guardrail evaluation metrics')
    parser.add_argument('--adapter-mode', type=str, choices=['single', 'merged'], default='single',
                      help='Adapter mode to plot (single or merged, default: single). This determines which sub-folder in behavior_reports is scanned.')
    parser.add_argument('--behaviors', type=str, nargs='+',
                      help='Specific behaviors to include in plot (e.g., politics expert_opinion). If not specified, attempts to plot all found for the adapter mode.')
    parser.add_argument('--output', type=str,
                      help='Output file name (default: plots/detection_efficiency_<mode>.png)')
    parser.add_argument('--list', action='store_true',
                      help='List available metrics sources and exit')
    args = parser.parse_args()
    
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # List available metrics sources if requested
    if args.list:
        print_available_metrics_new_format(args.adapter_mode)
        return
    
    # Load metrics data from scoring reports
    metrics_by_behavior = load_metrics_from_scoring_reports(args.adapter_mode, args.behaviors)
    
    if not metrics_by_behavior:
        print("No metrics found to plot based on the provided parameters and available reports.")
        print("Please ensure evaluation scripts have run and generated '*_scoring.txt' files in 'code/guardrail_evaluation/behavior_reports/<adapter_mode>/'")
        print("And 'politics_guardrail_report_llama_guard.txt' exists for Llama Guard data.")
        # Call the new listing function to guide the user
        print_available_metrics_new_format(args.adapter_mode)
        return
    
    # Determine output file
    if args.output:
        output_file = args.output
    else: 
        output_filename_mode = args.adapter_mode if args.adapter_mode else "single" 
        output_file = f"plots/detection_efficiency_{output_filename_mode}.png"
    
    # Print loaded behaviors
    print(f"Generating plot with behaviors: {', '.join(metrics_by_behavior.keys())}")
    
    # Generate plot
    plot_detection_efficiency(metrics_by_behavior, output_file)

if __name__ == "__main__":
    main() 