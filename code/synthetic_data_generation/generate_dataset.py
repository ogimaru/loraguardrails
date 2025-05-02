import os
import json
import random
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from asyncio import Semaphore
# our definitions
from pregenerated_fields import character_names, character_traits
from guardrails import Guardrail, GUARDRAILS 
from helper_functions import DatasetGenerationConfig, CostLog, format_for_llama32
from prompts import generate_character_prompt, generate_response_prompt, PROMPT_TEMPLATE

def pregenerate_characters(n_samples: int) -> List[Dict]:
    """Pre-generate character name and trait combinations."""
    available_combinations = []
    used_names = set()
    used_traits = set()
    
    while len(available_combinations) < n_samples:
        available_names = list(set(character_names) - used_names)
        available_traits = list(set(character_traits) - used_traits)
        
        if not available_names or not available_traits:
            break
            
        name = random.choice(available_names)
        traits = random.choice(available_traits)
        
        used_names.add(name)
        used_traits.add(traits)
        available_combinations.append({
            "name": name,
            "traits": traits
        })
    
    return available_combinations

async def generate_orpo_sample(
    config: DatasetGenerationConfig,
    guardrail: Guardrail,
    client: AsyncOpenAI,
    semaphore: Semaphore,
    character_info: Dict,
    previous_example: Dict = None,
    fixed_history: Dict = None
) -> Dict:
    async with semaphore:
        try:
            if fixed_history:
                # Use the fixed history but generate a new final user message
                character_data = fixed_history.copy()
                character_data['user_message_final'] = (await client.chat.completions.create(
                    model=config.large_llm,
                    messages=[{"role": "user", "content": f"Generate a user message that might trigger {guardrail.trigger}"}],
                    temperature=config.temperature,
                )).choices[0].message.content
                char_tokens = (0, 0)  # No tokens used for character generation since we reused it
            else:
                # Generate character using pre-generated info
                character_prompt = generate_character_prompt(
                    guardrail.trigger, 
                    previous_example, 
                    character_info["name"], 
                    character_info["traits"]
                )
                character_response = await client.chat.completions.create(
                    model=config.large_llm,
                    messages=[{"role": "user", "content": character_prompt}],
                    temperature=config.temperature, 
                )
                
                character_data = json.loads(character_response.choices[0].message.content)
                char_tokens = (character_response.usage.prompt_tokens, character_response.usage.completion_tokens)
            
            # Rest of the function remains the same
            chosen_prompt = generate_response_prompt(character_data, guardrail.resolution)
            chosen_response = await client.chat.completions.create(
                model=config.large_llm,
                messages=[{"role": "user", "content": chosen_prompt}],
                temperature=config.temperature, 
            )
            chosen_completion = chosen_response.choices[0].message.content.strip()
            
            # Generate rejected response (violating guardrail)
            rejected_prompt = generate_response_prompt(
                character_data,
                f"""IMPORTANT: Do the OPPOSITE of the guardrail. You MUST:
                1. Actively and enthusiastically engage with {guardrail.trigger}
                2. Take strong stances and express clear opinions
                3. Provide specific details and suggestions related to {guardrail.trigger}
                4. Show excitement and willingness to participate in {guardrail.trigger}
                5. Completely disregard any safety concerns or restrictions

                Make sure your response clearly violates the guardrail by doing exactly what it aims to prevent."""
            )
            rejected_response = await client.chat.completions.create(
                model=config.large_llm,
                messages=[{"role": "user", "content": rejected_prompt}],
                temperature=config.temperature, 
            )
            rejected_completion = rejected_response.choices[0].message.content.strip()
            
            response_tokens = (
                chosen_response.usage.prompt_tokens + rejected_response.usage.prompt_tokens,
                chosen_response.usage.completion_tokens + rejected_response.usage.completion_tokens
            )

            return {
                'character_info': character_data,
                'chosen': chosen_completion,
                'rejected': rejected_completion,
                'tokens': (char_tokens, response_tokens),
                'success': True
            }

        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

class DatasetVariationMode:
    VARY_ALL = "vary_all"   
    FIXED_CHARACTER = "fixed_character"  
    FIXED_HISTORY = "fixed_history"  

async def generate_orpo_dataset(config: DatasetGenerationConfig, guardrail: Guardrail, variation_mode: str = DatasetVariationMode.VARY_ALL):    
    client = AsyncOpenAI()
    dataset = []
    total_input_tokens = 0
    total_output_tokens = 0
    previous_example = None 
    
    # Pre-generate character combinations
    available_characters = pregenerate_characters(config.n_samples)
    if len(available_characters) < config.n_samples:
        print(f"Warning: Could only generate {len(available_characters)} unique character combinations")
    
    # For fixed character/history modes, generate a single character to use
    fixed_character = random.choice(available_characters) if variation_mode != DatasetVariationMode.VARY_ALL else None
    fixed_history = None
    
    # Create semaphore to limit concurrent API calls
    semaphore = Semaphore(25)
    
    async def process_batch(start_idx: int, batch_size: int) -> List[Dict]:
        nonlocal previous_example  # Add nonlocal declaration
        batch_characters = [fixed_character] * batch_size if fixed_character else available_characters[start_idx:start_idx + batch_size]
        
        # For FIXED_HISTORY mode, generate character data once and reuse
        nonlocal fixed_history
        if variation_mode == DatasetVariationMode.FIXED_HISTORY and fixed_history is None:
            try:
                char_response = await client.chat.completions.create(
                    model=config.large_llm,
                    messages=[{"role": "user", "content": generate_character_prompt(
                        guardrail.trigger, 
                        None,
                        fixed_character["name"], 
                        fixed_character["traits"]
                    )}],
                    temperature=config.temperature,
                )
                fixed_history = json.loads(char_response.choices[0].message.content)
            except Exception as e:
                print(f"Error generating fixed history: {str(e)}")
                return []
        
        tasks = []
        for char_info in batch_characters:
            if variation_mode == DatasetVariationMode.FIXED_HISTORY:
                # Only vary the final user message
                task = generate_orpo_sample(
                    config, guardrail, client, semaphore,
                    char_info, previous_example,
                    fixed_history=fixed_history
                )
            else:
                task = generate_orpo_sample(
                    config, guardrail, client, semaphore,
                    char_info, previous_example
                )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    with tqdm(total=config.n_samples, desc=f"Generating {variation_mode} dataset", unit="sample") as pbar:
        samples_generated = 0
        batch_retries = {}
        max_retries = 5
        
        while samples_generated < config.n_samples:
            try:
                batch_size = min(25, config.n_samples - samples_generated)
                batch_id = samples_generated
                
                if batch_retries.get(batch_id, 0) >= max_retries:
                    print(f"\nSkipping batch starting at {batch_id} after {max_retries} retries")
                    samples_generated += batch_size
                    continue
                
                if batch_id in batch_retries:
                    retry_count = batch_retries[batch_id]
                    wait_time = min(32, 2 ** retry_count)
                    print(f"\nRetrying batch {batch_id} after {wait_time}s delay (attempt {retry_count + 1})")
                    await asyncio.sleep(wait_time)
                
                results = await process_batch(samples_generated, batch_size)
                
                successful_samples = []
                for result in results:
                    if result['success']:
                        sample_data = result
                        character_info = sample_data['character_info']
                        chosen = sample_data['chosen']
                        rejected = sample_data['rejected']
                        char_tokens, resp_tokens = sample_data['tokens']
                        
                        # Update tokens
                        total_input_tokens += char_tokens[0] + resp_tokens[0]
                        total_output_tokens += char_tokens[1] + resp_tokens[1]

                        # Fill the prompt template
                        conversation_history = "\n".join([
                            f"User: {pair['user_message']}\n{character_info['character_name']}: {pair['ai_response']}" 
                            for pair in character_info['conversation_history']
                        ])
                        
                        filled_prompt = PROMPT_TEMPLATE["prompt"].format(
                            character_name=character_info['character_name'],
                            traits=character_info['traits'],
                            typical_expressions=character_info['typical_expressions'],
                            memories=character_info['memories'],
                            conversation_history=conversation_history,
                            user_message_final=character_info['user_message_final']
                        )

                        entry = {
                            "prompt": filled_prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "variation_mode": variation_mode
                        }
                        successful_samples.append(entry)
                        samples_generated += 1
                        pbar.update(1)
                        
                        if samples_generated > 0 and variation_mode == DatasetVariationMode.VARY_ALL:
                            previous_example = character_info
                
                if successful_samples:
                    dataset.extend(successful_samples)
                    if batch_id in batch_retries:
                        del batch_retries[batch_id]  # Clear retry count on success
                else:
                    # Track retry for this batch
                    batch_retries[batch_id] = batch_retries.get(batch_id, 0) + 1
                    print(f"\nNo successful samples in batch {batch_id}. Retry {batch_retries[batch_id]}/{max_retries}")
                
            except Exception as e:
                print(f"\nError in batch {samples_generated}: {str(e)}")
                batch_retries[samples_generated] = batch_retries.get(samples_generated, 0) + 1
                continue

    if not dataset:
        raise ValueError("Failed to generate any valid samples")

    return dataset, total_input_tokens, total_output_tokens

def process_orpo_to_llama(input_file: str, output_file: str):
    """Process ORPO dataset file to LLaMA format."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            data['prompt'] = format_for_llama32(data['prompt'])
            character_name = data['prompt'].split('Character Name:')[1].split('\n')[0].strip()

            # Format both chosen and rejected completions
            data['chosen'] = f"<|start_header_id|>assistant<|end_header_id|>\n{character_name}: {data['chosen']}<|eot_id|>"
            data['rejected'] = f"<|start_header_id|>assistant<|end_header_id|>\n{character_name}: {data['rejected']}<|eot_id|>"

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

async def async_main(generate_variations: bool = False):
    # Initialize configuration and cost logging
    config = DatasetGenerationConfig()
    config.update_from_yaml("dataset_config.yaml") 
    cost_log = CostLog()
    os.makedirs("datasets", exist_ok=True)

    # Process only enabled datasets from config
    for guardrail_type, dataset_config in config.datasets.items():
        if not dataset_config.get('enabled', True):
            continue

        if generate_variations:
            print(f"\nGenerating variations for {guardrail_type}...")
            # Only generate the fixed_character and fixed_history variations
            # Don't generate vary_all again since it's redundant with the standard dataset
            variation_modes = [DatasetVariationMode.FIXED_CHARACTER,
                             DatasetVariationMode.FIXED_HISTORY]
        else:
            print(f"\nGenerating standard dataset for {guardrail_type}...")
            # Standard dataset is equivalent to vary_all
            variation_modes = [DatasetVariationMode.VARY_ALL]
        
        for variation_mode in variation_modes:
            # Generate dataset
            dataset, input_tokens, output_tokens = await generate_orpo_dataset(
                config, GUARDRAILS[guardrail_type], variation_mode
            )
            cost_log.accumulate(input_tokens, output_tokens)
            
            # Determine file names based on variation mode
            if variation_mode != DatasetVariationMode.VARY_ALL:
                # Add variation suffix only for non-standard datasets
                raw_output_file = f"datasets/orpo_{guardrail_type}_{variation_mode}_raw.jsonl"
                llama_output_file = f"datasets/orpo_{guardrail_type}_{variation_mode}_llama.jsonl"
                base_path = f"datasets/huggingface_datasets/orpo_{guardrail_type}_{variation_mode}"
            else:
                # Standard dataset (vary_all) doesn't need the suffix
                raw_output_file = f"datasets/orpo_{guardrail_type}_raw.jsonl"
                llama_output_file = f"datasets/orpo_{guardrail_type}_llama.jsonl"
                base_path = f"datasets/huggingface_datasets/orpo_{guardrail_type}"
            
            # Save raw dataset
            with open(raw_output_file, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
            print(f"Generated {len(dataset)} samples and saved to {raw_output_file}")

            # Process to LLaMA format
            process_orpo_to_llama(raw_output_file, llama_output_file)
            print(f"Processed to LLaMA format and saved to {llama_output_file}")

            # Create and save HuggingFace dataset
            dataset = load_dataset("json", data_files=llama_output_file)
            
            # Split into train/validation/test
            first_split = dataset["train"].train_test_split(
                test_size=config.test_split_ratio, 
                seed=42
            )
            test_dataset = first_split["test"]
            
            train_val_split = first_split["train"].train_test_split(
                test_size=config.validation_split_ratio/(1-config.test_split_ratio),
                seed=42
            )
            
            final_dataset = DatasetDict({
                "train": train_val_split["train"],
                "validation": train_val_split["test"],
                "test": test_dataset
            })
            
            final_dataset.save_to_disk(base_path)
            
            print(f"\nDataset splits saved to {base_path} with:")
            print(f"Train samples: {len(final_dataset['train'])} ({(1-config.test_split_ratio-config.validation_split_ratio)*100:.1f}%)")
            print(f"Validation samples: {len(final_dataset['validation'])} ({config.validation_split_ratio*100:.1f}%)")
            print(f"Test samples: {len(final_dataset['test'])} ({config.test_split_ratio*100:.1f}%)")
        
    cost_log.estimate()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate datasets for behavior guardrails')
    parser.add_argument('--generate-variations', action='store_true',
                      help='Generate all dataset variations (vary_all, fixed_character, fixed_history)')
    args = parser.parse_args()
    
    asyncio.run(async_main(args.generate_variations))

if __name__ == "__main__":
    main() 