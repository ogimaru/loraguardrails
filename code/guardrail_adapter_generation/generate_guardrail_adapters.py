from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
import os
from datasets import load_from_disk
from trl import ORPOTrainer, ORPOConfig 
from unsloth import is_bfloat16_supported
from dataclasses import dataclass
from typing import List, Optional
import yaml
import pandas as pd
from transformers.trainer_callback import TrainerCallback
from tensorboard import program
import wandb

# Set wandb to offline mode to disable syncing while still storing reports locally
os.environ["WANDB_MODE"] = "offline" 
print("wandb configured to run in offline mode. Reports will be stored locally only.")

# Increase CUDA memory allocation for better performance
torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 256  # Increased from 128 for better performance

@dataclass
class ORPOLoRAConfig:
    # Base model parameters
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    load_in_4bit: bool = True

    # PEFT model parameters
    r: int = 16
    target_modules: List[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj")
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = True
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None

    # ORPO specific parameters
    beta: float = 0.1
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None

    # Training parameters
    validation_split: float = 0.2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    num_train_epochs: int = 50
    learning_rate: float = 2e-4
    dataset_num_proc: int = 2
    packing: bool = False
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"
    remove_unused_columns: bool = False

    # Output and storage parameters
    output_dir: str = "orpo_lora_adapters"
    adapter_type: str = None  # Single adapter type instead of a list

    # Evaluation parameters
    eval_strategy: str = "steps"
    eval_steps: float = 0.1
    save_strategy: str = "steps"
    save_steps: float = 0.1
    eval_batch_size: int = 16
    max_eval_samples: Optional[int] = None

    # Wandb configuration
    wandb_project: str = "orpo-lora-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = None
    wandb_api_key: Optional[str] = None

    # New dataset variation parameter
    dataset_variation: str = "vary_all"

    def update_from_yaml(self, yaml_filename: str):
        """Update config from YAML file."""
        if not os.path.exists(yaml_filename):
            raise FileNotFoundError(f"Config file {yaml_filename} not found")
            
        with open(yaml_filename, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        for key, value in yaml_config.items():
            if hasattr(self, key):
                # Convert numeric values to proper types
                field_type = type(getattr(self, key))
                if field_type in (float, int) and value is not None:
                    try:
                        value = field_type(value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Failed to convert {key}={value} to {field_type.__name__}: {str(e)}")
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' in YAML file is not a valid configuration parameter")

def validate_dataset(dataset, name):
    """Validate dataset has required fields for ORPO training."""
    required_fields = ["prompt", "chosen", "rejected"]
    
    for field in required_fields:
        if field not in dataset.features:
            raise ValueError(f"Dataset {name} is missing required field: {field}")
            
    if len(dataset) > 0:
        example = dataset[0]
        print(f"\nExample from {name}:")
        print("Prompt:", example["prompt"][:200] + "..." if len(example["prompt"]) > 200 else example["prompt"])
        print("Chosen:", example["chosen"][:200] + "..." if len(example["chosen"]) > 200 else example["chosen"])
        print("Rejected:", example["rejected"][:200] + "..." if len(example["rejected"]) > 200 else example["rejected"])
    return True

class CustomORPOTrainer(ORPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, is_training=True, num_items_in_batch=None):
        """Custom loss computation that tracks both full ORPO loss and its components."""
        # Compute forward pass and get components
        forward_output = self.concatenated_forward(model, inputs)
        policy_chosen_logps, policy_rejected_logps = forward_output[:2]
        policy_nll_loss = forward_output[4]
        
        # Get odds ratio loss components
        ratio_losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_chosen_logps, policy_rejected_logps
        )
        
        # Calculate both components
        preference_loss = -ratio_losses.mean()
        nll_loss = policy_nll_loss
        full_loss = nll_loss + preference_loss
        
        if is_training:
            loss = full_loss
            if return_outputs:
                metrics = {
                    "loss": loss.item(),
                    "nll_loss": nll_loss.item(),
                    "preference_loss": preference_loss.item(),
                    "rewards/chosen": chosen_rewards.mean(),
                    "rewards/rejected": rejected_rewards.mean(),
                    "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
                    "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
                    "logps/rejected": policy_rejected_logps.mean(),
                    "logps/chosen": policy_chosen_logps.mean(),
                    "log_odds_ratio": log_odds_ratio,
                    "log_odds_chosen": log_odds_chosen,
                }
                return loss, metrics
            return loss
        else:
            eval_loss = full_loss
            if return_outputs:
                metrics = {
                    "eval_loss": eval_loss.item(),
                    "eval_nll_loss": nll_loss.item(),
                    "eval_preference_loss": preference_loss.item(),
                    "eval_rewards/chosen": chosen_rewards.mean(),
                    "eval_rewards/rejected": rejected_rewards.mean(),
                    "eval_rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
                    "eval_rewards/margins": (chosen_rewards - rejected_rewards).mean(),
                    "eval_logps/rejected": policy_rejected_logps.mean(),
                    "eval_logps/chosen": policy_chosen_logps.mean(),
                    "eval_log_odds_ratio": log_odds_ratio,
                    "eval_log_odds_chosen": log_odds_chosen,
                }
                return eval_loss, metrics
            return eval_loss

def clean_gpu_memory():
    """Clean up GPU memory before starting training."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Force a sync point
    torch.cuda.synchronize()
    # Print memory stats
    print("\nGPU Memory Status Before Training:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

def train_and_save_orpo_lora(dataset_name: str, output_dir: str, config: ORPOLoRAConfig):
    """Train and save ORPO LoRA adapter using a local dataset."""
    try:
        # Clean GPU memory before starting
        clean_gpu_memory()
        
        # Initialize wandb if enabled
        if config.report_to == "wandb":
            # Set wandb to offline mode
            os.environ["WANDB_MODE"] = "offline"
            print("Setting wandb to offline mode. Reports will be stored locally only.")
            
            if config.wandb_api_key:
                os.environ["WANDB_API_KEY"] = config.wandb_api_key
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name or f"orpo-{config.model_name}-{dataset_name}",
                tags=config.wandb_tags,
                mode="offline"  # Set to offline mode to disable syncing
            )

        # Set up model
        print(f"Loading base model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
            device_map="auto",
            attn_implementation="flash_attention_2",
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": config.dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )

        model_with_lora = FastLanguageModel.get_peft_model(
            model,
            r=config.r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=config.random_state,
            use_rslora=config.use_rslora,
            loftq_config=config.loftq_config,
        )

        # Enable reward modelling stats
        PatchDPOTrainer()

        # Load and validate dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "..", "synthetic_data_generation", "datasets", "huggingface_datasets")
        
        if not os.path.exists(datasets_dir):
            raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

        # Check for dataset variation
        dataset_variation = config.dataset_variation if hasattr(config, "dataset_variation") else "vary_all"
        variation_suffix = ""
        
        # Determine dataset suffix based on variation
        if dataset_variation == "fixed_character":
            variation_suffix = "_fixed_character"
        elif dataset_variation == "fixed_history":
            variation_suffix = "_fixed_history"
        # vary_all has no suffix
        
        dataset_name = f"orpo_{config.adapter_type.replace('-', '_')}{variation_suffix}"
        dataset_path = os.path.join(datasets_dir, dataset_name)
        
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset not found for adapter type: {config.adapter_type} with variation: {dataset_variation}")

        print(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        validate_dataset(train_dataset, f"{dataset_name} (train)")
        validate_dataset(eval_dataset, f"{dataset_name} (validation)")

        print(f"\nDataset splits:")
        print(f"Training examples: {len(train_dataset)}")
        print(f"Validation examples: {len(eval_dataset)}")

        # Set up training
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, "runs")
        os.makedirs(tensorboard_dir, exist_ok=True)

        if config.report_to == "tensorboard":
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', tensorboard_dir])
            url = tb.launch()
            print(f"\nTensorBoard running at: {url}")

        # Calculate steps
        total_steps = len(train_dataset) * config.num_train_epochs / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        actual_eval_steps = max(1, int(total_steps * config.eval_steps))
        actual_save_steps = max(1, int(total_steps * config.save_steps))

        print(f"\nTraining Configuration:")
        print(f"Total steps: {total_steps}")
        print(f"Evaluation every {actual_eval_steps} steps")
        print(f"Save every {actual_save_steps} steps")

        # Set up trainer
        trainer = CustomORPOTrainer(
            model=model_with_lora,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=ORPOConfig(
                max_length=config.max_seq_length,
                max_prompt_length=config.max_prompt_length or config.max_seq_length//2,
                max_completion_length=config.max_completion_length or config.max_seq_length//2,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.eval_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                num_train_epochs=config.num_train_epochs,
                learning_rate=config.learning_rate,
                beta=config.beta,
                fp16=not config.bf16,
                bf16=config.bf16,
                logging_steps=config.logging_steps,
                optim=config.optim,
                weight_decay=config.weight_decay,
                lr_scheduler_type=config.lr_scheduler_type,
                seed=config.seed,
                output_dir=output_dir,
                eval_strategy=config.eval_strategy,
                eval_steps=actual_eval_steps,
                save_strategy=config.save_strategy,
                save_steps=actual_save_steps,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                report_to=[config.report_to] if config.report_to != "none" else [],
                logging_dir=tensorboard_dir,
                remove_unused_columns=config.remove_unused_columns,
            ),
        )

        # Add memory usage callback
        class MemoryCallback(TrainerCallback):
            """Callback to monitor GPU memory usage during training."""
            def __init__(self):
                self.peak_memory = 0
                self.last_memory = 0
                self.step_memory_increases = []
                self.last_logged_step = -1  # Track the last step we logged
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                
                # Get current GPU memory usage
                current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
                memory_increase = current_memory - self.last_memory
                
                # Update peak memory
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Track memory increase for this step
                if memory_increase > 0:
                    self.step_memory_increases.append(memory_increase)
                
                # Add memory usage to logs
                logs["gpu_memory_gb"] = current_memory
                logs["peak_memory_gb"] = self.peak_memory
                
                # Only print memory usage once per step to avoid duplication
                if state.global_step % 10 == 0 and state.global_step != self.last_logged_step:
                    self.last_logged_step = state.global_step
                    print(f"\nMemory Usage at step {state.global_step}:")
                    print(f"  Current: {current_memory:.2f} GB")
                    print(f"  Peak: {self.peak_memory:.2f} GB")
                    print(f"  Increase: {memory_increase:.2f} GB")
                    
                    # Calculate average memory increase per step
                    if self.step_memory_increases:
                        avg_increase = sum(self.step_memory_increases) / len(self.step_memory_increases)
                        print(f"  Avg Increase/Step: {avg_increase:.4f} GB")
                    
                    # Calculate free memory
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
                    free_memory = total_memory - current_memory
                    print(f"  Free Memory: {free_memory:.2f} GB / {total_memory:.2f} GB")
                
                # Update last memory
                self.last_memory = current_memory

        trainer.add_callback(MemoryCallback())

        # Train and save
        trainer_stats = trainer.train()
        model_with_lora.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save metrics
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_df = pd.DataFrame(trainer.state.log_history)
        # Save metrics to CSV with adapter type
        metrics_df.to_csv(os.path.join(metrics_dir, f"training_metrics_{config.adapter_type}.csv"), index=False)

        if wandb.run is not None:
            wandb.finish()

        return trainer_stats

    except Exception as e:
        print(f"Error training on {dataset_name}: {str(e)}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
        torch.cuda.empty_cache()

def main():  
    # Set wandb to offline mode globally
    os.environ["WANDB_MODE"] = "offline"
    print("Setting wandb to offline mode globally. All reports will be stored locally only.")
    
    # Load config
    config = ORPOLoRAConfig()
    config.update_from_yaml("adapter_config.yaml")
    os.makedirs(config.output_dir, exist_ok=True)

    # Validate adapter type
    if not config.adapter_type:
        raise ValueError("adapter_type must be specified in the config file")
    
    valid_adapter_types = ["politics", "expert_opinion", "meeting"]
    if config.adapter_type not in valid_adapter_types:
        raise ValueError(f"Invalid adapter_type: {config.adapter_type}. Must be one of {valid_adapter_types}")

    # Get dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "..", "synthetic_data_generation", "datasets", "huggingface_datasets")
    
    if not os.path.exists(datasets_dir):
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    # Check for dataset variation
    dataset_variation = config.dataset_variation if hasattr(config, "dataset_variation") else "vary_all"
    variation_suffix = ""
    
    # Determine dataset suffix based on variation
    if dataset_variation == "fixed_character":
        variation_suffix = "_fixed_character"
    elif dataset_variation == "fixed_history":
        variation_suffix = "_fixed_history" 
    
    dataset_name = f"orpo_{config.adapter_type.replace('-', '_')}{variation_suffix}"
    dataset_path = os.path.join(datasets_dir, dataset_name)
    
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset not found for adapter type: {config.adapter_type} with variation: {dataset_variation}")

    print(f"Processing dataset: {dataset_name} (variation: {dataset_variation})")
    output_dir = os.path.join(config.output_dir, f"orpo_model_{config.adapter_type}_{dataset_variation}")
    
    trainer_stats = train_and_save_orpo_lora(dataset_name, output_dir, config)
    
    print(f"\nResults for {dataset_name}:")
    print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    print(f"Peak memory: {torch.cuda.max_memory_reserved()/1024/1024/1024:.2f} GB")

if __name__ == "__main__":
    main() 