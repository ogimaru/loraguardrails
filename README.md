# Behavioral Guardrails for Dynamic LLM Persona - Code Implementation + Manuscript

This directory contains the implementation code for the Behavioral LoRA Guardrail system. This README provides technical details on how to use and extend the codebase.

## Directory Structure

```
code/
├── requirements.txt                # Main dependencies
├── synthetic_data_generation/      # Data generation for adapter training
├── guardrail_adapter_generation/   # LoRA adapter training with ORPO
├── guardrail_evaluation/           # Evaluation of trained adapters
└── plot_generation/                # Visualization of results
```

## Environment Setup

We recommend using a Python virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For best performance, also install flash-attention
pip install flash-attn
```

The workflow will use gpt4o-mini as the frontier LLM alongside Llama Guard hosted by together.ai. Currently the setup requires the API keys to be configured as environment variables.

For Windows (PowerShell):
```powershell
# Set OpenAI API key for gpt4-mini
$env:OPENAI_API_KEY = "your-openai-api-key"

# Set Together API key for LlamaGuard
$env:TOGETHER_API_KEY = "your-together-api-key"
```

For Windows (Command Prompt):
```cmd
# Set OpenAI API key for gpt4-mini
set OPENAI_API_KEY=your-openai-api-key

# Set Together API key for LlamaGuard
set TOGETHER_API_KEY=your-together-api-key
```

For Linux:
```bash
# Set OpenAI API key for gpt4-mini
export OPENAI_API_KEY="your-openai-api-key"

# Set Together API key for LlamaGuard
export TOGETHER_API_KEY="your-together-api-key"
```

Replace `your-openai-api-key` and `your-together-api-key` with your actual API keys. Make sure to keep these keys secure and never commit them to version control.

## End-to-End Workflow

### 0. Using the Master Pipeline Script

For convenience, we've created a master script that can run the entire pipeline from start to finish:

```bash
# Run the entire pipeline
python run_full_pipeline.py

# Run specific steps of the pipeline (1: Generate datasets, 2: Train adapters, 3: Evaluate, 4: Generate plots)
python run_full_pipeline.py --start-step 2 --end-step 3  # Only run steps 2-3
```

The master script performs these tasks in sequence:
1. **Generate synthetic datasets**
   - Creates vary_all datasets for all behavior types (politics, expert_opinion, meeting)
   - Creates additional variations (fixed_character, fixed_history) only for politics
   
2. **Train LoRA adapters**
   - Trains adapters for all behavior types with vary_all dataset variation
   - Trains politics adapter with fixed_character and fixed_history variations
   
3. **Evaluate adapters**
   - Evaluates all adapters trained on vary_all dataset
   - Evaluates politics adapters trained on fixed_character and fixed_history variations
   - Compares performance across all variations for politics
   - Runs LlamaGuard comparison for politics guardrail
   - Evaluates merged adapters using SVD merging
   - Evaluates merged adapters using linear merging 
   
4. **Generate plots**
   - Creates training metrics plots
   - Creates detection efficiency plots

The pipeline manages directory changes, GPU memory cleanup, and appropriate wait times between steps.

If you prefer to run individual components separately, follow the detailed steps below.

### 1. Generate Training Data

Prepare the config file.
Key configuration options in `dataset_config.yaml`:
- `n_samples`: Number of samples to generate
- `behaviors`: List of behaviors to generate data for  

```bash
cd synthetic_data_generation
python generate_dataset.py
```

The dataset will be saved in the `datasets` directory.

### 2. Train LoRA Adapters

Training configuration (`adapter_config.yaml`):
- Base model parameters:
  - `model_name`: Base model to use (default: "unsloth/Meta-Llama-3.1-8B-Instruct")
  - `max_seq_length`: Maximum sequence length
  - `load_in_4bit`/`load_in_8bit`: Quantization for memory efficiency
- LoRA parameters:
  - `r`: LoRA rank (default: 16)
  - `target_modules`: Layers to apply LoRA (all projection layers)
  - `lora_alpha`: LoRA alpha parameter
- ORPO specific parameters:
  - `beta`: Relative ratio loss weight (default: 0.1)
- Training parameters:
  - `per_device_train_batch_size`: Batch size per GPU
  - `gradient_accumulation_steps`: Accumulation steps
  - `num_train_epochs`: Number of training epochs
  - `learning_rate`: Learning rate for optimization
  - `eval_steps`: Evaluation frequency
  - `save_steps`: Checkpoint saving frequency
- WandB configuration:
  - `wandb_project`: Project name for tracking
  - `wandb_entity`: Optional entity name
  - `wandb_run_name`: Optional run name
  - `wandb_tags`: Optional tags

For a single behavior:
```bash
cd ../guardrail_adapter_generation
python generate_guardrail_adapters.py
```

For multiple behavior adapters:
```bash
# Default: Generate all behavior types with vary_all variation
python generate_multiple_adapters.py

# Generate a specific behavior (with default vary_all variation)
python generate_multiple_adapters.py --behavior-type politics

# Generate all behaviors with a specific variation
python generate_multiple_adapters.py --dataset-variation fixed_character

# Generate a specific behavior with a specific variation
python generate_multiple_adapters.py --behavior-type politics --dataset-variation fixed_character

# Generate a specific behavior with all variations
python generate_multiple_adapters.py --behavior-type politics --dataset-variation all

# Generate all behaviors with all variations (generates all combinations)
python generate_multiple_adapters.py --behavior-type all --dataset-variation all
```

For faster training on machines with more GPU memory:
```yaml
per_device_train_batch_size: 8  # Increase this
gradient_accumulation_steps: 1  # Decrease this
load_in_8bit: true              # Use 8-bit instead of 4-bit for better speed
use_gradient_checkpointing: false # Disable if you have enough memory
```

### 3. Evaluate Adapters

For running evaluation on all behaviors:
```bash
cd ../guardrail_evaluation
python run_all_behaviors_single.py 
```

For running evaluation for the adapter trained on different dataset variations:
```bash
python run_all_variations.py
```

The evaluation generates:
1. Metrics files in `../plot_generation/metrics/` 
2. Human-readable reports in `behavior_reports/`

To test a specific behavior:
```bash
python run_all_behaviors_single.py --behavior politics
```

For more detailed debugging:
```bash
python testing_lora_adapters.py --adapter-path ../guardrail_adapter_generation/orpo_lora_adapters/orpo_model_politics_vary_all --behavior politics --debug
```

For comparing with LlamaGuard (need togetherai api key):
```bash
python testing_politics_guardrail_with_llama_guard.py
```

### 4. Generate Plots

```bash
cd ../plot_generation
python detection_efficiency_plot.py
python make_training_plot.py
```

Plots are saved to the `plots/` directory, with configuration options in `plot_config.yaml`.

## Technical Implementation Details

### Guardrail Implementation

Currently, the system implements three primary guardrails:

1. **Politics Guardrail**: Prevents engagement in political discussions
2. **Expert Opinion Guardrail**: Avoids providing specialized expert advice
3. **Meeting Guardrail**: Prevents attempts to meet the AI avatar in person

Each guardrail consists of a trigger description and resolution instructions. These are defined in `synthetic_data_generation/guardrails.py`.

### ORPO Training

We use ORPO (Odds Ratio Preference Optimization) to generate the LoRA adapters in `guardrail_adapter_generation/generate_guardrail_adapters.py`. 
The key components are:

1. **Dataset Preparation**: Organizing preferred and rejected examples
2. **Model Quantization**: Using 4-bit quantization to reduce memory usage 
4. **Weights & Biases Integration**: Experimental tracking in offline mode
5. **Memory Optimization**: Enhanced memory management with synchronization points

The training objective optimizes:
- Maximizing likelihood for preferred responses
- Minimizing likelihood for rejected responses

### Evaluation Methodology

The evaluation in `guardrail_evaluation/testing_lora_adapters.py` measures:

1. **Behavior Detection**: Testing if the model identifies target behaviors
2. **Natural Preservation**: Testing if the model preserves non-target behaviors
3. **False Positive Analysis**: Measuring incorrect classifications
4. **Adapter Merging**: Improved support for combining multiple guardrail adapters

The evaluation results include:
- Detailed reports in the `behavior_reports/` directory
- Summary metrics in JSON format for visualization
- Comprehensive metrics that show:
  - Overall accuracy
  - Behavior-specific accuracy
  - Natural prompt handling accuracy
  - Guard tag usage percentage

When comparing variations with `run_all_variations.py`, the output provides:
- A summary table with columns for behavior type, variation, adapter mode, and all metrics
- A composite score ranking that combines overall accuracy, behavior accuracy, and natural handling
- CSV output for further analysis in `variation_comparison_results.csv`

### Key Implementation Notes

1. **Compatibility Issues**: `save_steps` must be a multiple of `eval_steps` when using `load_best_model_at_end=true`

2. **Memory Management**: 
   - 8-bit quantization is faster than 4-bit but uses more memory
   - Gradient checkpointing trades speed for memory efficiency
   - Flash attention accelerates training on supported hardware
   - Custom memory tracking provides insight into GPU usage patterns

3. **Metrics File Naming Convention**:
   Each behavior's metrics are saved as:
   ```
   metrics_{behavior}_{adapter_mode}_{variation}.json
   ```
   Example: `metrics_politics_single_vary_all.json`

4. **Training Metrics**:
   Training metrics are saved during adapter generation at:
   ```
   guardrail_adapter_generation/orpo_lora_adapters/orpo_model_{behavior}_{variation}/metrics/training_metrics_{behavior}.csv
   ```
   
5. **Weights & Biases Integration**:
   - Now operates in offline mode by default (`WANDB_MODE=offline`)
   - Reports are stored locally without requiring internet connectivity
   - Training runs can be analyzed offline using WandB local UI

## Extending the Codebase

### Adding a New Behavior

1. Define the behavior in `synthetic_data_generation/guardrails.py`:
   ```python
   GUARDRAILS = {
       # Existing guardrails...
       "new_behavior": Guardrail(
           name="new_behavior",
           trigger="Description of the problematic behavior to handle.", 
           resolution="How the model should resolve the issue."
       )
   }
   ```

2. Generate data for the new behavior:
   ```bash
   cd synthetic_data_generation
   python generate_dataset.py --behavior new_behavior
   ```

3. Train an adapter for the new behavior:
   ```bash
   cd ../guardrail_adapter_generation
   python generate_guardrail_adapters.py --adapter-type new_behavior
   ```

4. Evaluate the new adapter:
   ```bash
   cd ../guardrail_evaluation
   python run_all_behaviors_single.py --behavior new_behavior
   ```

### Using Checkpoints for Evaluation

During adapter training, checkpoints are saved at regular intervals (controlled by `save_steps` in `adapter_config.yaml`). You can evaluate these intermediate checkpoints to compare their performance:

1. To evaluate using a specific checkpoint, update the `checkpoint_path` in `testing_config.yaml`:
   ```yaml
   # Use a specific checkpoint (example: checkpoint-250)
   checkpoint_path: "checkpoint-250"
   
   # Or use the main adapter weights (final model)
   checkpoint_path: ""
   ```

2. Alternatively, specify the checkpoint directly when running evaluation:
   ```bash
   python testing_lora_adapters.py --behavior politics --checkpoint-path checkpoint-250
   ```

This allows you to analyze how the adapter's performance evolves during training and identify the optimal checkpoint.

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter CUDA out-of-memory errors:
1. Reduce batch size: `per_device_train_batch_size: 1`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Enable 4-bit quantization: `load_in_4bit: true`
4. Enable gradient checkpointing: `use_gradient_checkpointing: true` 

### Training Issues

- **Incompatible eval_steps and save_steps**:
  Ensure `save_steps` is a multiple of `eval_steps`:
  ```yaml
  eval_steps: 12
  save_steps: 24  # 24 is a multiple of 12
  ``` 

- **WandB Issues**:
  - WandB now operates in offline mode by default
  - If you want to sync to the cloud, set `WANDB_MODE=online` in your environment

### Missing Metrics Files

If metrics files are missing after evaluation:
1. Check the output path in `testing_lora_adapters.py`
2. Verify that the behavior name matches exactly
3. Check console output for any file saving errors

## Contact

For issues, questions, or contributions, please submit an issue or pull request to the repository. 