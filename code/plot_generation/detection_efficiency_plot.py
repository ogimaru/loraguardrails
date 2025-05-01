import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import glob
from typing import List

def find_metrics_files():
    """Find available metrics files in the metrics directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.join(script_dir, "metrics")
    
    # Check if directory exists
    if not os.path.exists(metrics_dir):
        print(f"Warning: Metrics directory not found at {metrics_dir}")
        return []
    
    # Get all JSON files in the metrics directory
    metrics_files = glob.glob(os.path.join(metrics_dir, "*.json"))
    return [os.path.basename(f) for f in metrics_files]

def print_available_metrics(metrics_files):
    """Print available metrics files."""
    if not metrics_files:
        print("No metrics files found in metrics/ directory.")
        print("Make sure to run evaluation scripts first to generate metrics files.")
        return
    
    print("\nAvailable metrics files:")
    for i, file in enumerate(metrics_files):
        print(f"  {i+1}. {file}")
        
        # Try to read file and show behaviors included
        try:
            with open(os.path.join("metrics", file), 'r') as f:
                data = json.load(f)
                behaviors = list(data.keys())
                print(f"     Behaviors: {', '.join(behaviors)}")
        except Exception as e:
            print(f"     (Could not read file contents: {str(e)})")

def load_behavior_metrics(adapter_mode: str = "single", behaviors: List[str] = None):
    """Load metrics for specified behaviors or all available behaviors."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.join(script_dir, "metrics")
    
    if not os.path.exists(metrics_dir):
        print(f"Warning: Metrics directory not found at {metrics_dir}")
        return {}
    
    # Find all behavior metrics files with the pattern metric_{behavior}_{adapter_mode}.json
    available_files = glob.glob(os.path.join(metrics_dir, f"metric_*_{adapter_mode}.json"))
    
    # Also look for LlamaGuard metrics
    llama_guard_file = os.path.join(metrics_dir, "metric_politics_llama_guard.json")
    
    # Extract behavior names from filenames
    available_behaviors = []
    for file in available_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1].endswith('.json'):
            # Extract behavior from filename (metric_behavior_adapter_mode.json)
            behavior = "_".join(parts[1:-1])
            available_behaviors.append(behavior)
    
    # Determine which behaviors to load
    if behaviors:
        target_behaviors = [b for b in behaviors if b in available_behaviors]
        missing = [b for b in behaviors if b not in available_behaviors]
        if missing:
            print(f"Warning: Metrics not found for behaviors: {', '.join(missing)}")
    else:
        target_behaviors = available_behaviors
    
    if not target_behaviors and not os.path.exists(llama_guard_file):
        print(f"No behaviors found with metrics files matching adapter mode: {adapter_mode}")
        return {}
    
    print(f"Loading metrics for behaviors: {', '.join(target_behaviors)}")
    
    # Load metrics for each behavior
    metrics_by_behavior = {}
    
    for behavior in target_behaviors:
        metrics_file = os.path.join(metrics_dir, f"metric_{behavior}_{adapter_mode}.json")
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    # Extract metrics for this behavior
                    if behavior in data:
                        metrics_by_behavior[behavior] = data[behavior]
                        print(f"Loaded metrics for {behavior} from {metrics_file}")
                    else:
                        # Try to use the first key if behavior not found directly
                        first_key = list(data.keys())[0]
                        metrics_by_behavior[behavior] = data[first_key]
                        print(f"Loaded metrics for {behavior} using key {first_key} from {metrics_file}")
            except Exception as e:
                print(f"Error loading metrics for {behavior}: {str(e)}")
    
    # Load LlamaGuard metrics if available
    if os.path.exists(llama_guard_file):
        try:
            with open(llama_guard_file, 'r') as f:
                llama_guard_data = json.load(f)
                if "politics_llama_guard" in llama_guard_data:
                    metrics_by_behavior["politics_llama_guard"] = llama_guard_data["politics_llama_guard"]
                    print("Loaded LlamaGuard metrics")
        except Exception as e:
            print(f"Error loading LlamaGuard metrics: {str(e)}")
    
    return metrics_by_behavior

def plot_detection_efficiency(metrics_by_behavior, output_file="plots/detection_efficiency.png"):
    """Create plots for detection efficiency using Seaborn."""
    # Set publication-quality plot parameters
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
    
    # Make the figure more compact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Define consistent colors
    palette = {
        'Baseline': 'royalblue',
        'With Guardrail': 'lightgreen',
        'LLama Guard': 'orange'
    }
    
    # First plot: Sets of two bars for each behavior type
    behaviors = [b for b in metrics_by_behavior.keys() if b != "politics_llama_guard"]
    
    # Prepare data for Seaborn - focus on behavior detection rates
    data = []
    for behavior in behaviors:
        # Get behavior detection rates (baseline = without_lora, guardrail = with_lora)
        baseline = metrics_by_behavior[behavior].get('behavior_detection_without_lora', 0) * 100
        guardrail = metrics_by_behavior[behavior].get('behavior_detection_with_lora', 0) * 100
        
        data.append({
            'Behavior': behavior.capitalize(),
            'Detection Rate': baseline,
            'Type': 'Baseline'
        })
        data.append({
            'Behavior': behavior.capitalize(),
            'Detection Rate': guardrail,
            'Type': 'With Guardrail'
        })
    
    df = pd.DataFrame(data)
    
    # Plot with Seaborn - with fixed bar width
    sns.barplot(x='Behavior', y='Detection Rate', hue='Type', data=df, palette=palette, ax=ax1, width=0.8)
    
    ax1.set_ylabel('Behavior Detection Rate (%)')
    ax1.set_title('')
    
    # Remove the legend from the left plot - we'll add it to the right plot
    ax1.get_legend().remove()
    
    # Remove the "Behavior" label from x-axis
    ax1.set_xlabel('')
    
    # Add values on top of bars - only if the height is significant
    for p in ax1.patches:
        if p.get_height() > 1.0:  # Only show labels for bars with height > 1%
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
        
        # Add both "Politics" bars to the SAME DataFrame we used for the left plot
        # This ensures absolutely identical treatment by seaborn
        comparison_data = [
            {
                'Behavior': 'Comparison',  # Using a new category name
                'Detection Rate': politics_detection,
                'Type': 'With Guardrail'
            },
            {
                'Behavior': 'Comparison',  # Using a new category name
                'Type': 'LLama Guard',
                'Detection Rate': llama_guard_detection
            }
        ]
        
        # Add the comparison data to the same DataFrame
        extended_df = pd.concat([df, pd.DataFrame(comparison_data)], ignore_index=True)
        
        # Use the same exact plotting approach for the right figure 
        comparison_df = extended_df[extended_df['Behavior'] == 'Comparison']
        sns.barplot(x='Behavior', y='Detection Rate', hue='Type', 
                   data=comparison_df,
                   palette={'With Guardrail': palette['With Guardrail'], 'LLama Guard': palette['LLama Guard']}, 
                   ax=ax2, width=0.8)
        
        # Move y-label to the right side
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylabel('')  # Remove redundant y-label on the right
        ax2.set_title('')
        ax2.set_xlabel('')
        
        # Set the x-label to "Politics" for clarity
        ax2.set_xticklabels(['Politics'])
        
        # Remove the legend from the right plot
        ax2.get_legend().remove()
        
        # Add values on top of bars - only if the height is significant
        for p in ax2.patches:
            if p.get_height() > 1.0:  # Only show labels for bars with height > 1%
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
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.15, 0.5), framealpha=0.9, fontsize=20)
    else:
        ax2.set_visible(False)
    
    # Adjust the spacing to accommodate the legend in the middle
    plt.subplots_adjust(wspace=0.1)  # Reduced from 0.2 to almost no space
    
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
    parser.add_argument('--data-file', type=str, 
                      help='JSON file containing metrics data (optional - will use individual behavior files if not specified)')
    parser.add_argument('--adapter-mode', type=str, choices=['single', 'merged'], default='single',
                      help='Adapter mode to plot (single or merged, default: single)')
    parser.add_argument('--behaviors', type=str, nargs='+',
                      help='Specific behaviors to include in plot (default: all available)')
    parser.add_argument('--output', type=str,
                      help='Output file name (default: plots/detection_efficiency_<mode>.png)')
    parser.add_argument('--list', action='store_true',
                      help='List available metrics files and exit')
    args = parser.parse_args()
    
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Find available metrics files
    metrics_files = find_metrics_files()
    
    # List available metrics files if requested
    if args.list:
        print_available_metrics(metrics_files)
        return
    
    # Load metrics data - either from specified file or from individual behavior files
    metrics_by_behavior = {}
    
    if args.data_file:
        # Use provided data file
        metrics_path = args.data_file
        if not os.path.exists(metrics_path) and not metrics_path.startswith("metrics/"):
            metrics_path = os.path.join("metrics", metrics_path)
        
        # Check if the file exists
        if not os.path.exists(metrics_path):
            print(f"Error: Metrics file not found at {metrics_path}")
            print("Available metrics files:")
            print_available_metrics(metrics_files)
            print("\nPlease run evaluation scripts or specify a different metrics file.")
            return
            
        # Load metrics data
        try:
            with open(metrics_path, 'r') as f:
                metrics_by_behavior = json.load(f)
            print(f"Loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"Error loading metrics from {metrics_path}: {str(e)}")
            return
    else:
        # Load from individual behavior metrics files
        metrics_by_behavior = load_behavior_metrics(args.adapter_mode, args.behaviors)
        
        if not metrics_by_behavior:
            print("No metrics found for the specified behaviors and adapter mode.")
            print("Available metrics files:")
            print_available_metrics(metrics_files)
            print("\nPlease run evaluation scripts or specify different parameters.")
            return
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        adapter_mode = args.adapter_mode if args.adapter_mode else "single"
        output_file = f"plots/detection_efficiency_{adapter_mode}.png"
    
    # Print loaded behaviors
    print(f"Generating plot with behaviors: {', '.join(metrics_by_behavior.keys())}")
    
    # Generate plot
    plot_detection_efficiency(metrics_by_behavior, output_file)

if __name__ == "__main__":
    main() 