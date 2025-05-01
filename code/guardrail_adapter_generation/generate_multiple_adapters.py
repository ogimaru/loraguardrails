#!/usr/bin/env python
import os
import sys
import yaml
import time
import subprocess
from pathlib import Path
import argparse
import torch

def clean_gpu_memory():
    """Clean up GPU memory between runs."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("\nGPU Memory Status After Cleanup:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

def update_config(config_file, behavior_type, dataset_variation):
    """Update the adapter_config.yaml file with specific behavior and dataset variation."""
    # Load existing config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update behavior type and dataset variation
    config['adapter_type'] = behavior_type
    config['dataset_variation'] = dataset_variation
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config: adapter_type={behavior_type}, dataset_variation={dataset_variation}")

def run_adapter_generation(script_path):
    """Run the adapter generation script as a subprocess."""
    try:
        print(f"\n{'='*80}")
        print(f"STARTING ADAPTER GENERATION")
        print(f"{'='*80}\n")
        
        # Use subprocess to run the generation script
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            print(f"\nProcess exited with error code {process.returncode}")
            return False
        else:
            print(f"\nProcess completed successfully")
            return True
            
    except Exception as e:
        print(f"Error running adapter generation: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate multiple adapters with various configurations')
    parser.add_argument('--behavior-type', type=str, default='all',
                        choices=['all', 'politics', 'expert_opinion', 'meeting'],
                        help='Behavior type(s) to generate adapters for. Default is "all" (generates all behaviors)')
    parser.add_argument('--dataset-variation', type=str, default='vary_all',
                        choices=['vary_all', 'fixed_character', 'fixed_history', 'all'],
                        help='Dataset variation to use. Default is "vary_all". Use "all" to generate all variations.')
    args = parser.parse_args()
    
    # Define behaviors and variations
    behaviors = ["politics", "expert_opinion", "meeting"]
    variations = ["vary_all", "fixed_character", "fixed_history"]
    
    # Determine which behaviors to process
    selected_behaviors = behaviors if args.behavior_type == 'all' else [args.behavior_type]
    selected_variations = variations if args.dataset_variation == 'all' else [args.dataset_variation]
    
    # Get the path to the configuration file
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_file = script_dir / "adapter_config.yaml"
    generator_script = script_dir / "generate_guardrail_adapters.py"
    
    # Validate paths
    if not config_file.exists():
        print(f"Error: Configuration file not found at {config_file}")
        return
    
    if not generator_script.exists():
        print(f"Error: Generator script not found at {generator_script}")
        return
    
    # Print configuration header
    print(f"\n\n{'='*100}")
    if len(selected_behaviors) > 1:
        print(f"= GENERATING ADAPTERS FOR ALL BEHAVIORS: {', '.join(selected_behaviors).upper()}")
    else:
        print(f"= GENERATING ADAPTER FOR: {selected_behaviors[0].upper()}")
    
    if len(selected_variations) > 1:
        print(f"= WITH ALL VARIATIONS: {', '.join(selected_variations).upper()}")
    else:
        print(f"= WITH VARIATION: {selected_variations[0].upper()}")
    print(f"{'='*100}\n")
    
    # Process each behavior with the selected variations
    for behavior in selected_behaviors:
        for variation in selected_variations:
            print(f"\n\n{'#'*100}")
            print(f"# GENERATING ADAPTER FOR: {behavior.upper()}")
            print(f"# Dataset variation: {variation}")
            print(f"{'#'*100}\n")
            
            update_config(config_file, behavior, variation)
            success = run_adapter_generation(generator_script)
            clean_gpu_memory()
            
            if not success:
                print(f"Warning: Generation failed for {behavior} with variation {variation}. Continuing with next combination.")
            
            if len(selected_behaviors) > 1 or len(selected_variations) > 1:
                print(f"\nWaiting 10 seconds before next generation...")
                time.sleep(10)
    
    # Print completion summary
    print(f"\n\n{'='*100}")
    print(f"ADAPTER GENERATION COMPLETE")
    if len(selected_behaviors) > 1:
        print(f"Generated adapters for: {', '.join(selected_behaviors)}")
    else:
        print(f"Generated adapter for: {selected_behaviors[0]}")
    
    if len(selected_variations) > 1:
        print(f"With variations: {', '.join(selected_variations)}")
    else:
        print(f"With variation: {selected_variations[0]}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main() 