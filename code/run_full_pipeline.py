#!/usr/bin/env python3
"""
Master script to run the full pipeline for behavior LoRA guardrail manuscript:
1. Generate synthetic datasets
2. Train LoRA adapters
3. Evaluate adapters
4. Generate plots

This script coordinates running all components in the correct sequence.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import argparse
import yaml
import shutil

def run_command(cmd, description):
    """Run a command and display its output in real-time."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
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
    
    # Wait for process to complete
    process.wait()
    
    success = process.returncode == 0
    if not success:
        print(f"\nWARNING: Process exited with error code {process.returncode}")
    
    return success

def wait_between_steps(seconds=10):
    """Wait between steps to allow resources to be freed."""
    print(f"\nWaiting {seconds} seconds before next step...")
    for i in range(seconds, 0, -1):
        print(f"\rContinuing in {i} seconds...", end='')
        time.sleep(1)
    print("\rContinuing now...            ")

def generate_datasets():
    """Generate all required datasets."""
    print("\n\n" + "="*100)
    print("STEP 1: GENERATING SYNTHETIC DATASETS")
    print("="*100)
    
    # Change to the synthetic_data_generation directory
    synthetic_data_dir = os.path.join(os.getcwd(), "synthetic_data_generation")
    os.chdir(synthetic_data_dir)
    
    # First handle dataset config to control which behaviors are enabled
    config_file = os.path.join(synthetic_data_dir, "dataset_config.yaml")
    
    # 1. Generate standard datasets for all behavior types (politics, expert_opinion, meeting) 
    print("\nGenerating standard datasets for all behavior types")
    success = run_command(
        [sys.executable, "generate_dataset.py"],
        "Generating standard datasets for all behavior types"
    )
    
    if not success:
        print("WARNING: Standard dataset generation failed.")
        return False
    
    wait_between_steps(5)
    
    # 2. Now temporarily modify config to enable only politics for additional variations
    print("\nPreparing to generate additional variations for politics only")
    
    # Backup the original config
    backup_config = config_file + ".backup"
    shutil.copy2(config_file, backup_config)
    print(f"Backed up original config to {backup_config}")
    
    try:
        # Read the current config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Disable all datasets except politics
        for behavior_type in config.get("datasets", {}):
            if behavior_type != "politics":
                if "enabled" in config["datasets"][behavior_type]:
                    config["datasets"][behavior_type]["enabled"] = False
                else:
                    config["datasets"][behavior_type]["enabled"] = False
        
        # Ensure politics is enabled
        if "politics" in config.get("datasets", {}):
            config["datasets"]["politics"]["enabled"] = True
        
        # Write the modified config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("Modified dataset_config.yaml to enable only politics behavior")
        
        # Now generate additional variations for politics only 
        print("\nGenerating fixed_character and fixed_history variations for politics only")
        success = run_command(
            [sys.executable, "generate_dataset.py", "--generate-variations"],
            "Generating fixed_character and fixed_history variations for politics only"
        )
        
        if not success:
            print("WARNING: Dataset generation failed for politics variations.")
        
    finally:
        # Restore the original config
        shutil.copy2(backup_config, config_file)
        os.remove(backup_config)
        print("Restored original dataset_config.yaml")
    
    # Return to the main code directory
    os.chdir("..")
    return True

def generate_adapters():
    """Generate all required adapters."""
    print("\n\n" + "="*100)
    print("STEP 2: TRAINING LORA ADAPTERS")
    print("="*100)
    
    # Change to the guardrail_adapter_generation directory
    adapter_dir = os.path.join(os.getcwd(), "guardrail_adapter_generation")
    os.chdir(adapter_dir)
    
    # 1. Generate all adapters for vary_all dataset variation
    print("\nGenerating all adapters for vary_all dataset variation")
    success = run_command(
        [sys.executable, "generate_multiple_adapters.py", "--behavior-type", "all", "--dataset-variation", "vary_all"],
        "Generating all behavior adapters with vary_all variation"
    )
    
    if not success:
        print("WARNING: Adapter generation failed for vary_all variation.")
    
    wait_between_steps(10)  # Longer wait after adapter generation to clear GPU memory
    
    # 2. Generate politics adapter for fixed_character and fixed_history variations
    print("\nGenerating politics adapters with fixed_character and fixed_history variations")
    success = run_command(
        [sys.executable, "generate_multiple_adapters.py", "--behavior-type", "politics", "--dataset-variation", "all"],
        "Generating politics adapters with all variations"
    )
    
    if not success:
        print("WARNING: Adapter generation failed for politics with variations.")
    
    # Return to the main code directory
    os.chdir("..")
    return True

def evaluate_adapters():
    """Evaluate all generated adapters."""
    print("\n\n" + "="*100)
    print("STEP 3: EVALUATING ADAPTERS")
    print("="*100)
    
    # Change to the guardrail_evaluation directory
    eval_dir = os.path.join(os.getcwd(), "guardrail_evaluation")
    os.chdir(eval_dir)
    
    # 1. Evaluate all adapters trained on vary_all dataset
    print("\nEvaluating all adapters trained on vary_all dataset")
    success = run_command(
        [sys.executable, "run_all_behaviors_single.py", "--variation", "vary_all"],
        "Evaluating all behavior adapters with vary_all variation"
    )
    
    if not success:
        print("WARNING: Evaluation failed for vary_all variation.")
    
    wait_between_steps(10)
    
    # 2. Evaluate politics adapters trained on fixed_character and fixed_history
    variations = ["fixed_character", "fixed_history"]
    for variation in variations:
        print(f"\nEvaluating politics adapter trained on {variation} dataset")
        success = run_command(
            [sys.executable, "run_all_behaviors_single.py", "--behavior", "politics", "--variation", variation],
            f"Evaluating politics adapter with {variation} variation"
        )
        
        if not success:
            print(f"WARNING: Evaluation failed for politics with {variation}.")
        
        wait_between_steps(10)
    
    # 3. Evaluate merged adapters using SVD merging
    print("\nEvaluating merged adapters using SVD merging")
    os.makedirs(os.path.join("behavior_reports", "merged"), exist_ok=True)
    
    success = run_command(
        [sys.executable, "testing_lora_adapters.py", "--variation", "vary_all", "--adapter-mode", "merged"],
        "Evaluating merged behavior adapters with SVD merging"
    )
    
    if not success:
        print("WARNING: Evaluation failed for SVD merged adapters.")
    
    wait_between_steps(10)
    
    # 4. Evaluate merged adapters using linear merging
    print("\nEvaluating merged adapters using linear merging")
    success = run_command(
        [sys.executable, "testing_lora_adapters.py", "--behavior", "politics", "--behavior", "expert_opinion", "--behavior", "meeting", "--variation", "vary_all", "--adapter-mode", "linear_merged"],
        "Evaluating merged behavior adapters with linear merging"
    )
    
    if not success:
        print("WARNING: Evaluation failed for linearly merged adapters.")
    
    wait_between_steps(10)
    
    # 5. Compare politics adapter across all variations
    print("\nComparing politics adapter across all variations")
    success = run_command(
        [sys.executable, "run_all_variations.py"],
        "Comparing politics adapter across all variations"
    )
    
    if not success:
        print("WARNING: Variation comparison failed.")
    
    wait_between_steps(10)
    
    # 5. Run Llama Guard evaluation for politics guardrail
    print("\nRunning Llama Guard evaluation for politics guardrail")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join("behavior_reports", "llama_guard"), exist_ok=True)
    
    # Run the Llama Guard evaluation only for the default (vary_all) variation
    print("\nRunning Llama Guard evaluation for politics guardrail with default (vary_all) variation")
    success = run_command(
        [sys.executable, "testing_politics_guardrail_with_llama_guard.py", "--variation", "vary_all"],
        "Evaluating politics guardrail with Llama Guard - default variation"
    )
    
    if not success:
        print("WARNING: Llama Guard evaluation failed.")
    
    # Return to the main code directory
    os.chdir("..")
    return True

def generate_plots():
    """Generate analysis plots from results."""
    print("\n\n" + "="*100)
    print("STEP 4: GENERATING PLOTS")
    print("="*100)
    
    # Change to the plot_generation directory
    plot_dir = os.path.join(os.getcwd(), "plot_generation")
    os.chdir(plot_dir)
    
    # 1. Generate training plots
    print("\nGenerating training plots")
    success = run_command(
        [sys.executable, "make_training_plot.py"],
        "Generating training metrics plots"
    )
    
    if not success:
        print("WARNING: Training plot generation failed.")
    
    wait_between_steps(5)
    
    # 2. Generate detection efficiency plots
    print("\nGenerating detection efficiency plots")
    success = run_command(
        [sys.executable, "detection_efficiency_plot.py"],
        "Generating detection efficiency plots"
    )
    
    if not success:
        print("WARNING: Detection efficiency plot generation failed.")
    
    # Return to the main code directory
    os.chdir("..")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the full behavior LoRA guardrail pipeline')
    parser.add_argument('--start-step', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Start at a specific step (1: Generate datasets, 2: Train adapters, 3: Evaluate, 4: Generate plots)')
    parser.add_argument('--end-step', type=int, default=4, choices=[1, 2, 3, 4],
                      help='End at a specific step (1: Generate datasets, 2: Train adapters, 3: Evaluate, 4: Generate plots)')
    args = parser.parse_args()
    
    # Ensure we're in the code directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Define the pipeline steps
    pipeline_steps = [
        (1, "Generate datasets", generate_datasets),
        (2, "Train adapters", generate_adapters),
        (3, "Evaluate adapters", evaluate_adapters),
        (4, "Generate plots", generate_plots)
    ]
    
    # Execute the pipeline
    print("\n" + "="*100)
    print(f"STARTING FULL PIPELINE (steps {args.start_step}-{args.end_step})")
    print("="*100)
    
    start_time = time.time()
    
    for step_num, step_name, step_func in pipeline_steps:
        if args.start_step <= step_num <= args.end_step:
            step_start = time.time()
            print(f"\nExecuting step {step_num}/{len(pipeline_steps)}: {step_name}")
            success = step_func()
            step_end = time.time()
            
            print(f"\nStep {step_num} completed in {(step_end - step_start)/60:.2f} minutes")
            if not success:
                print(f"WARNING: Step {step_num} encountered issues but continuing with pipeline")
            
            if step_num < args.end_step:
                wait_between_steps(10)
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print("\n" + "="*100)
    print(f"PIPELINE COMPLETED IN {total_time:.2f} MINUTES")
    print("="*100)

if __name__ == "__main__":
    main() 
    