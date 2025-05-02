import os
import json
import yaml
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class CostLog:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0
    def accumulate(self, add_input_tokens, add_output_tokens):
        self.total_input_tokens += add_input_tokens
        self.total_output_tokens += add_output_tokens
        self.estimated_cost = self.estimate_cost()

    def estimate_cost(self):
        # NOTE: Quickly outdated prices. 
        input_price_per_1m_tokens = 0.15 
        output_price_per_1m_tokens = 0.60  
        input_cost = (self.total_input_tokens / 1_000_000) * input_price_per_1m_tokens
        output_cost = (self.total_output_tokens / 1_000_000) * output_price_per_1m_tokens
        return input_cost + output_cost

    def estimate(self):
        print(f"Total input tokens (all datasets): {self.total_input_tokens}\nTotal output tokens (all datasets): {self.total_output_tokens}\nEstimated total cost: ${self.estimated_cost:.6f}")

@dataclass
class DatasetGenerationConfig:
    n_samples: int = 100 
    train_split_ratio: float = 0.8  # Default to 80% training data
    test_split_ratio: float = 0.1   # Default to 10% test data
    validation_split_ratio: float = 0.1  # Default to 10% validation data
    large_llm: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 100
    datasets: dict = None

    def update_from_yaml(self, yaml_filename: str):
        """Update config from YAML file, only for fields that are present in the YAML."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, yaml_filename)
        
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            for key, value in yaml_config.items():
                setattr(self, key, value)
            
            # Update sample sizes if tuning size changes
            if 'n_samples' in yaml_config:
                self.n_samples_tuning = int(self.n_samples * self.train_split_ratio)
                self.n_samples_testing = int(self.n_samples * self.test_split_ratio)
                self.n_samples_evaluation = int(self.n_samples * self.validation_split_ratio)
            
            # Validate split ratios sum to 1
            total_split = self.train_split_ratio + self.test_split_ratio + self.validation_split_ratio
            if not (0.99 <= total_split <= 1.01):  # Allow for small floating point errors
                raise ValueError(f"Split ratios must sum to 1, got {total_split}")
                
        except FileNotFoundError:
            print(f"Warning: Config file {yaml_path} not found. Using default values.")
        except Exception as e:
            print(f"Error reading config file: {e}")

def format_for_llama32(prompt_text, completion=None):
    """Format text for LLaMA 3.2 model with appropriate headers."""
    # Split the prompt to separate system instructions and conversation
    parts = prompt_text.split('Conversation History:', 1)
    system_instructions = parts[0].strip()
    conversation = parts[1].strip() if len(parts) > 1 else ""

    # Get character name from system instructions
    character_name = system_instructions.split('Character Name:')[1].split('\n')[0].strip()

    # Format the system part
    formatted_prompt = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n"
    formatted_prompt += system_instructions + "\n\nConversation History:\n"
    
    # Format conversation history
    conversation_lines = [line.strip() for line in conversation.split('\n') if line.strip()]
    last_user_message = None
    
    # Find the last user message
    for line in conversation_lines:
        if line.startswith('User:'):
            last_user_message = line

    # Process all messages except the last user message
    for line in conversation_lines:
        if line.startswith('User:'):
            if line != last_user_message:  # Skip if it's the last user message
                formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n"
                formatted_prompt += f"{line}<|eot_id|>\n"
        elif line.startswith(f"{character_name}:"):
            if not line.endswith(':'):  # Skip empty assistant messages
                formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
                formatted_prompt += f"{line}<|eot_id|>\n"

    # Add the final user message
    formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n"
    formatted_prompt += f"{last_user_message}<|eot_id|>"

    return formatted_prompt

def process_file_LLaMA(input_file, output_file):
    """Process a file to format it for LLaMA 3.2 model."""
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                data['prompt'] = format_for_llama32(data['prompt'])

                # Handle completion if it exists
                if 'completion' in data:
                    character_name = data['prompt'].split('Character Name:')[1].split('\n')[0].strip()
                    data['completion'] = f"<|start_header_id|>assistant<|end_header_id|>{character_name}: {data['completion']}<|eot_id|>"

                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

        print(f"Processing complete. Updated data written to {output_file}")
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")

def convert_and_store_dataset(train_jsonl, test_jsonl, eval_jsonl, dataset_name):
    """Convert training, testing, and evaluation JSONL files to Huggingface dataset format and store locally."""
    dataset_dict = load_dataset('json', data_files={
        'train': train_jsonl,
        'test': test_jsonl,
        'evaluation': eval_jsonl
    })
    
    os.makedirs("datasets/huggingface_datasets", exist_ok=True)
    dataset_dict.save_to_disk(f"datasets/huggingface_datasets/{dataset_name}")
    
    print(f"Dataset successfully stored in: datasets/huggingface_datasets/{dataset_name}")
    print(f"Train split size: {len(dataset_dict['train'])}, Test split size: {len(dataset_dict['test'])}, Evaluation split size: {len(dataset_dict['evaluation'])}")