#!/usr/bin/env python3
"""
Script to run testing_lora_adapters.py for all behaviors sequentially.
This helps manage memory by only loading one adapter at a time.
"""

import os
import subprocess
import time
import sys
import yaml
import json
from typing import List, Dict
import argparse
import glob

def load_behaviors_from_config(config_file: str) -> List[str]:
    """Load the behaviors list from the testing config file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('wanted_behaviors', [])
    except Exception as e:
        print(f"Error loading config file {config_file}: {str(e)}")
        return []

def run_evaluation_for_behavior(behavior: str, variation: str = "vary_all"):
    """Run the testing_lora_adapters.py script for a specific behavior."""
    print(f"\n{'='*80}")
    print(f"EVALUATING BEHAVIOR: {behavior}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "testing_lora_adapters.py",
        "--behavior", behavior,
        "--variation", variation,
        "--adapter-mode", "single"  # Explicitly set single adapter mode
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running evaluation for {behavior}. Return code: {process.returncode}")
            return False
        
        print(f"\nSuccessfully completed evaluation for {behavior}")
        
        # Verify the metrics file was created
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..")
        metrics_dir = os.path.join(project_root, "plot_generation", "metrics")
        metrics_file = os.path.join(metrics_dir, f"metric_{behavior}_single.json")
        
        if os.path.exists(metrics_file):
            print(f"Verified: Metrics file created at {metrics_file}")
            return True
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            return False
        
    except Exception as e:
        print(f"Exception when running evaluation for {behavior}: {str(e)}")
        return False

def wait_for_gpu_cooldown(seconds: int = 10):
    """Wait for the GPU to cool down and memory to be cleared."""
    print(f"\nWaiting for {seconds} seconds to ensure GPU memory is cleared...")
    time.sleep(seconds)
    print("Continuing with next behavior...")

def check_behavior_metrics(behaviors: List[str], adapter_mode: str = "single"):
    """Verify that metrics files exist for each behavior."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    metrics_dir = os.path.join(project_root, "plot_generation", "metrics")
    
    print(f"\nChecking metrics files for behaviors:")
    
    # Create an index file that lists all available behaviors and their metrics files
    metrics_index = {}
    
    for behavior in behaviors:
        metrics_file = os.path.join(metrics_dir, f"metric_{behavior}_{adapter_mode}.json")
        print(f"  - {behavior}: ", end="")
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    if behavior in data:
                        metrics_index[behavior] = os.path.basename(metrics_file)
                        print(f"OK - {metrics_file}")
                    else:
                        print(f"ERROR - File exists but missing {behavior} key")
            except Exception as e:
                print(f"ERROR - Could not read metrics file: {str(e)}")
        else:
            print(f"MISSING - {metrics_file}")
    
    # Write index file if metrics were found
    if metrics_index:
        index_file = os.path.join(metrics_dir, f"metrics_index_{adapter_mode}.json")
        with open(index_file, 'w') as f:
            json.dump({
                "adapter_mode": adapter_mode,
                "behaviors": metrics_index,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        print(f"\nMetrics index saved to: {index_file}")
        
    print(f"\nMetrics check complete. Files are stored in: {metrics_dir}")
    return len(metrics_index) > 0

def main():
    parser = argparse.ArgumentParser(description='Run testing_lora_adapters.py for all behaviors sequentially')
    parser.add_argument('--variation', type=str, default='vary_all', 
                      help='Dataset variation to use (vary_all, fixed_character, fixed_history)')
    parser.add_argument('--config', type=str, default='testing_config.yaml',
                      help='Path to testing config file')
    parser.add_argument('--cooldown', type=int, default=10,
                      help='Seconds to wait between behaviors for GPU memory to clear')
    parser.add_argument('--behavior', type=str,
                      help='Specific behavior to run')
    parser.add_argument('--behaviors', type=str, nargs='+',
                      help='Multiple specific behaviors to run (overrides config file)')
    args = parser.parse_args()
    
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Using dataset variation: {args.variation}")
    
    # Load behaviors from config or use provided behaviors
    if args.behavior:
        behaviors = [args.behavior]
        print(f"Using single behavior: {behaviors}")
    elif args.behaviors:
        behaviors = args.behaviors
        print(f"Using provided behaviors: {behaviors}")
    else:
        behaviors = load_behaviors_from_config(args.config)
        print(f"Loaded behaviors from config: {behaviors}")
    
    if not behaviors:
        print("No behaviors found to evaluate. Please check your config file or provide --behavior/--behaviors.")
        return
    
    # Run evaluation for each behavior
    results = {}
    for i, behavior in enumerate(behaviors):
        print(f"\nProcessing behavior {i+1}/{len(behaviors)}: {behavior}")
        success = run_evaluation_for_behavior(behavior, args.variation)
        results[behavior] = "Success" if success else "Failed"
        
        # Wait for GPU to cool down between behaviors (except for the last one)
        if i < len(behaviors) - 1:
            wait_for_gpu_cooldown(args.cooldown)
    
    # Check that metrics files were created for each behavior
    check_behavior_metrics(behaviors)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Behavior':<15} | {'Status':<10}")
    print(f"{'-'*15}-+-{'-'*10}")
    
    for behavior, status in results.items():
        print(f"{behavior:<15} | {status:<10}")
    
    print(f"\nEvaluation complete. Individual metrics files have been generated in the plot_generation/metrics directory.")

if __name__ == "__main__":
    main() 