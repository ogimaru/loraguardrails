import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob
from typing import List 
import matplotlib.ticker as ticker

def find_adapter_metrics_files(behaviors=["politics", "expert_opinion", "meeting"], 
                             variation="vary_all"):
    """Find training metrics files in the adapter directories for specified behaviors."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_base_dir = os.path.join(script_dir, "..", "guardrail_adapter_generation", "orpo_lora_adapters")
    
    if not os.path.exists(adapter_base_dir):
        print(f"Warning: Adapter directory not found at {adapter_base_dir}")
        return []
    
    metrics_files = []
     
    for behavior in behaviors: 
        found = False
        
        # Try different variations
        for variation_name in [variation, "fixed_history", "fixed_character", "vary_all"]:
            if found:
                break
                
            adapter_dir = os.path.join(adapter_base_dir, f"orpo_model_{behavior}_{variation_name}")
            
            if not os.path.exists(adapter_dir):
                continue
                    
            # Try different filename patterns
            possible_filenames = [
                f"training_metrics_{behavior}.csv",   
                "training_metrics.csv"                
            ]
            
            for filename in possible_filenames:
                metrics_file = os.path.join(adapter_dir, "metrics", filename)
                
                if os.path.exists(metrics_file):
                    metrics_files.append((behavior, metrics_file))
                    found = True
                    print(f"Found metrics for {behavior} at: {metrics_file}")
                    break
        
        # If still not found, try without variation suffix
        if not found:
            adapter_dir = os.path.join(adapter_base_dir, f"orpo_model_{behavior}")
            
            if os.path.exists(adapter_dir):
                # Try different filename patterns again
                possible_filenames = [
                    f"training_metrics_{behavior}.csv",   
                    "training_metrics.csv"                
                ]
                
                for filename in possible_filenames:
                    metrics_file = os.path.join(adapter_dir, "metrics", filename)
                    
                    if os.path.exists(metrics_file):
                        metrics_files.append((behavior, metrics_file))
                        found = True
                        print(f"Found metrics for {behavior} at: {metrics_file}")
                        break
    
    return metrics_files

def plot_training_metrics(metrics_files: List[str], dataset_names: List[str]):
    """
    Create three sets of plots side by side, each containing training loss and evaluation metrics.
    Optimized for publication-quality figures.
    
    Args:
        metrics_files (list): List of paths to three metrics CSV files
        dataset_names (list): List of names for each dataset (e.g., "Politics", "Expert opinions")
    """
    if len(metrics_files) != 3 or len(dataset_names) != 3:
        raise ValueError("Must provide exactly three metrics files and three dataset names")
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'figure.figsize': (20, 12),      
        'font.family': 'serif',
        'font.size': 28,                 
        'axes.titlesize': 32,            
        'axes.labelsize': 28,            
        'xtick.labelsize': 24,           
        'ytick.labelsize': 24,           
        'legend.fontsize': 24,           
        'lines.linewidth': 3.0,          
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.grid': True,  
        'axes.grid.which': 'major',
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
     
    fig, axes = plt.subplots(2, 3)
     
    train_loss_max = float('-inf')
    train_loss_min = float('inf')
    eval_metrics_max = float('-inf')
    eval_metrics_min = float('inf')
    
    # First pass to find global limits
    for col, (metrics_file, dataset_name) in enumerate(zip(metrics_files, dataset_names)):
        df = pd.read_csv(metrics_file)
        train_df = df[df['loss'].notna()].copy() if 'loss' in df else df[df['train_loss'].notna()].copy()
        eval_df = df[df['eval_nll_loss'].notna()].copy()
        
        if len(train_df) > 0:
            loss_col = 'loss' if 'loss' in train_df else 'train_loss'
            train_loss_max = max(train_loss_max, train_df[loss_col].max())
            train_loss_min = min(train_loss_min, train_df[loss_col].min())
        
        if len(eval_df) > 0:
            eval_metrics = ['eval_nll_loss', 'eval_log_odds_ratio']
            for metric in eval_metrics:
                if metric in eval_df:
                    eval_metrics_max = max(eval_metrics_max, eval_df[metric].max())
                    eval_metrics_min = min(eval_metrics_min, eval_df[metric].min())
    
    # Add some padding to the limits
    train_loss_padding = (train_loss_max - train_loss_min) * 0.1
    eval_metrics_padding = (eval_metrics_max - eval_metrics_min) * 0.1

    # Ensure minimum y value is 0 for evaluation metrics
    eval_metrics_min = 0
    
    # Plot for each column using different metrics files
    for col, (metrics_file, dataset_name) in enumerate(zip(metrics_files, dataset_names)):
        # Read the metrics file
        df = pd.read_csv(metrics_file)
        
        # Handle the alternating row structure in the CSV
        train_df = df[df['loss'].notna()].copy() if 'loss' in df else df[df['train_loss'].notna()].copy()
        eval_df = df[df['eval_nll_loss'].notna()].copy()
        
        # Plot 1: Training Loss (top row)
        if len(train_df) > 0:
            loss_col = 'loss' if 'loss' in train_df else 'train_loss'
            axes[0, col].plot(train_df['step'], train_df[loss_col], 
                            color='#1f77b4', label='Training Loss', alpha=0.8)
            axes[0, col].set_ylim(train_loss_min - train_loss_padding, 
                                train_loss_max + train_loss_padding)
            
            # Add minor ticks at 0.2 intervals (reduced from 0.1)
            axes[0, col].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            axes[0, col].tick_params(axis='y', which='minor', length=4)
            axes[0, col].grid(which='minor', alpha=0.2)
        
        # Plot 2: Evaluation Metrics (bottom row)
        if len(eval_df) > 0:
            axes[1, col].plot(eval_df['step'], eval_df['eval_nll_loss'], 
                            color='#d62728', label='NLL Loss', alpha=0.8, linestyle='-.') 
            
            preference_loss = -eval_df['eval_log_odds_ratio'] 
            axes[1, col].plot(eval_df['step'], preference_loss, 
                                color='#2ca02c', label='Preference Loss', alpha=0.8, linestyle='--') 
            
            # Calculate and plot total loss with solid line
            total_loss = eval_df['eval_nll_loss'] + (-eval_df['eval_log_odds_ratio'])
            axes[1, col].plot(eval_df['step'], total_loss,
                            color='#1f77b4', label='Total Loss', alpha=0.8, linestyle='-')
            
            axes[1, col].set_ylim(eval_metrics_min - eval_metrics_padding,
                                eval_metrics_max + eval_metrics_padding)
        
        # Set labels and titles
        if col == 0:
            axes[0, col].set_ylabel('Training Loss')
            axes[1, col].set_ylabel('Evaluation Metrics')
        elif col == 1:
            axes[0, col].set_ylabel('')
            axes[1, col].set_ylabel('')
            axes[0, col].set_yticklabels([])
            axes[1, col].set_yticklabels([])
            axes[0, col].tick_params(axis='y', which='both', length=0) 
            axes[1, col].tick_params(axis='y', which='both', length=0) 
            axes[0, col].spines.left.set_visible(False)
            axes[1, col].spines.left.set_visible(False)
        elif col == 2:
            axes[0, col].yaxis.set_label_position("right")
            axes[1, col].yaxis.set_label_position("right")
            axes[0, col].yaxis.tick_right()
            axes[1, col].yaxis.tick_right()
            axes[0, col].set_ylabel('')  
            axes[1, col].set_ylabel('')  
            axes[0, col].spines.left.set_visible(False)
            axes[1, col].spines.left.set_visible(False)
            axes[0, col].spines.right.set_visible(True)
            axes[1, col].spines.right.set_visible(True)
        
        axes[0, col].set_title(dataset_name)
        axes[0, col].legend(loc='upper right')
        axes[1, col].set_xlabel('Steps')
        axes[1, col].legend(loc='upper right')
        axes[0, col].set_xticklabels([])
    
    # Adjust layout to make the plot more compact
    plt.subplots_adjust(wspace=0.01, hspace=0.2)
    
    plt.tight_layout(pad=0.4)
    
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/training_eval_metrics_comparison.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.05)
    print(f"Plot saved to plots/training_eval_metrics_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find metrics files in the adapter directories
    print("Looking for training metrics in the adapter directories...")
    behaviors = ["politics", "expert_opinion", "meeting"]
    found_metrics = find_adapter_metrics_files(behaviors, variation="vary_all")
    
    if not found_metrics or len(found_metrics) < 3:
        print(f"Warning: Not enough training metrics files found ({len(found_metrics)}/3 needed)")
        print("Please make sure the adapter directories contain metrics files at:")
        print("  orpo_lora_adapters/orpo_model_<behavior>_<variation>/metrics/training_metrics_<behavior>.csv")
        print("  or")
        print("  orpo_lora_adapters/orpo_model_<behavior>_<variation>/metrics/training_metrics.csv")
        
        # List what was found
        if found_metrics:
            print("\nFound the following training metrics:")
            for behavior, path in found_metrics:
                print(f"  {behavior}: {path}")
        
        # Explain which ones are missing
        found_behaviors = [b for b, _ in found_metrics]
        missing_behaviors = [b for b in behaviors if b not in found_behaviors]
        if missing_behaviors:
            print(f"\nMissing training metrics for: {', '.join(missing_behaviors)}")
        
        # Try to find adapters directly without specific behaviors
        adapter_base_dir = os.path.join(script_dir, "..", "guardrail_adapter_generation", "orpo_lora_adapters")
        all_adapters = glob.glob(os.path.join(adapter_base_dir, "orpo_model_*"))
        all_metrics = []
        
        for adapter_dir in all_adapters:
            adapter_name = os.path.basename(adapter_dir)
            
            # Extract behavior from adapter_name (orpo_model_politics_vary_all -> politics)
            if "_" in adapter_name:
                parts = adapter_name.split("_")
                if len(parts) >= 3:
                    behavior = parts[2]  # Extract behavior from adapter name
                    
                    # Try both filename patterns
                    metrics_file = os.path.join(adapter_dir, "metrics", f"training_metrics_{behavior}.csv")
                    if os.path.exists(metrics_file):
                        all_metrics.append((behavior, metrics_file))
                        continue
            
            # Fallback to generic filename
            metrics_file = os.path.join(adapter_dir, "metrics", "training_metrics.csv")
            if os.path.exists(metrics_file):
                # Try to extract behavior from directory name
                adapter_name = os.path.basename(adapter_dir)
                if "_" in adapter_name:
                    parts = adapter_name.split("_")
                    if len(parts) >= 3:
                        behavior = parts[2]  # Extract behavior from adapter name
                    else:
                        behavior = "unknown"
                else:
                    behavior = "unknown"
                
                all_metrics.append((behavior, metrics_file))
        
        if all_metrics:
            print("\nFound other adapter metrics that might work:")
            for behavior, path in all_metrics:
                print(f"  {behavior}: {path}")
            
            # If we have enough metrics files, use them instead
            if len(all_metrics) >= 3:
                print("\nUsing the first 3 available metrics files found.")
                found_metrics = all_metrics[:3]
        
        if len(found_metrics) < 3:
            print("\nError: Cannot create plot without 3 metrics files.")
            exit(1)
    
    # Extract dataset names and metrics files
    dataset_names = []
    metrics_files = []
    
    for behavior, metrics_file in found_metrics[:3]:  # Take the first 3 if more are found 
        dataset_name = behavior.replace('_', ' ').title()
        dataset_names.append(dataset_name)
        metrics_files.append(metrics_file)
    
    print(f"\nGenerating plot with metrics from: {', '.join(dataset_names)}")
    plot_training_metrics(metrics_files, dataset_names)
