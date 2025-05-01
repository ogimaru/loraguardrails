import numpy as np
from dataclasses import dataclass, field
from typing import List, TextIO 
import os


@dataclass
class EvaluationItem:
    prompt: str
    response_with_lora: str
    response_without_lora: str
    is_natural: bool
    behavior_description: str

@dataclass
class EvaluationResult:
    correct_with_lora: bool
    correct_without_lora: bool
    explanation_with_lora: str
    explanation_without_lora: str
    evaluation_with_lora: str
    evaluation_without_lora: str

@dataclass
class TestingResults:
    accuracy_with_lora: float = 0.0
    accuracy_without_lora: float = 0.0
    accuracy_increase: float = 0.0
    behavior_accuracy_with_lora: float = 0.0
    behavior_accuracy_without_lora: float = 0.0
    natural_accuracy_with_lora: float = 0.0
    natural_accuracy_without_lora: float = 0.0 
    total_responses: int = 0
    total_behavior_responses: int = 0
    total_natural_responses: int = 0 
    correct_responses_with_lora: int = 0
    correct_responses_without_lora: int = 0
    correct_behavior_with_lora: int = 0
    correct_behavior_without_lora: int = 0
    correct_natural_with_lora: int = 0
    correct_natural_without_lora: int = 0
    guard_tags_with_lora: int = 0

    def update_results(self, eval_result: EvaluationResult):
        self.correct_responses_with_lora += eval_result.correct_with_lora
        self.correct_responses_without_lora += eval_result.correct_without_lora
        self.correct_behavior_with_lora += eval_result.correct_behavior_with_lora
        self.correct_behavior_without_lora += eval_result.correct_behavior_without_lora
        self.correct_natural_with_lora += eval_result.correct_natural_with_lora
        self.correct_natural_without_lora += eval_result.correct_natural_without_lora

def write_evaluation_to_report(report_file, index: int, eval_item: EvaluationItem, eval_result: EvaluationResult):
    """Write evaluation results to the report file."""
    if eval_item.is_natural:
        report_file.write(f"\n=== Example {index + 1} (natural) ===\n")
    else:
        report_file.write(f"\n=== Example {index + 1} (guardrail) ===\n")
    report_file.write(f"Prompt:\n{eval_item.prompt}\n\n")
    
    report_file.write("=== With LoRA ===\n")
    report_file.write(f"Response: {eval_item.response_with_lora}\n")
    report_file.write(f"Correct: {eval_result.correct_with_lora}\n")
    report_file.write(f"Explanation: {eval_result.explanation_with_lora}\n")
    report_file.write(f"Evaluation: {eval_result.evaluation_with_lora}\n\n")
    
    report_file.write("=== Without LoRA ===\n")
    report_file.write(f"Response: {eval_item.response_without_lora}\n")
    report_file.write(f"Correct: {eval_result.correct_without_lora}\n")
    report_file.write(f"Explanation: {eval_result.explanation_without_lora}\n")
    report_file.write(f"Evaluation: {eval_result.evaluation_without_lora}\n")
    report_file.write("\n" + "="*50 + "\n")



@dataclass
class EvaluationCosts:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0
    def accumulate_costs(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.estimated_cost = self.estimate_cost()
    
    def estimate_cost(self):
        input_price_per_1m_tokens = 0.15  # $0.15 per 1M input tokens
        output_price_per_1m_tokens = 0.60  # $0.60 per 1M output tokens
        
        input_cost = (self.total_input_tokens / 1_000_000) * input_price_per_1m_tokens
        output_cost = (self.total_output_tokens / 1_000_000) * output_price_per_1m_tokens
        
        return input_cost + output_cost
    
    def print_evaluation_costs(self) -> None:
        print("\nEvaluation Cost Summary:")
        print(f"Total input tokens: {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        print(f"Estimated total cost: ${self.estimated_cost:.6f}")


@dataclass 
class AccuracyResults:
    accuracy_with_lora: List[float] = field(default_factory=list)
    accuracy_without_lora: List[float] = field(default_factory=list)
    behavior_accuracy_with_lora: List[float] = field(default_factory=list)
    behavior_accuracy_without_lora: List[float] = field(default_factory=list)
    natural_accuracy_with_lora: List[float] = field(default_factory=list)
    natural_accuracy_without_lora: List[float] = field(default_factory=list)
    guard_tags_percentage: List[float] = field(default_factory=list)

    def update_accuracy_results(self, testing_results: TestingResults, verbose: bool = False):
        if verbose:
            print(f"\nDEBUG: Updating accuracy results with values:")
            print(f"Accuracy with LoRA: {testing_results.accuracy_with_lora:.2f}%")
            print(f"Accuracy without LoRA: {testing_results.accuracy_without_lora:.2f}%")
            print(f"Behavior accuracy with LoRA: {testing_results.behavior_accuracy_with_lora:.2f}%")
            print(f"Behavior accuracy without LoRA: {testing_results.behavior_accuracy_without_lora:.2f}%")
            print(f"Natural accuracy with LoRA: {testing_results.natural_accuracy_with_lora:.2f}%")
            print(f"Natural accuracy without LoRA: {testing_results.natural_accuracy_without_lora:.2f}%\n")
        
        self.accuracy_with_lora.append(testing_results.accuracy_with_lora)
        self.accuracy_without_lora.append(testing_results.accuracy_without_lora)
        self.behavior_accuracy_with_lora.append(testing_results.behavior_accuracy_with_lora)
        self.behavior_accuracy_without_lora.append(testing_results.behavior_accuracy_without_lora)
        self.natural_accuracy_with_lora.append(testing_results.natural_accuracy_with_lora)
        self.natural_accuracy_without_lora.append(testing_results.natural_accuracy_without_lora)
        if testing_results.total_behavior_responses > 0:
            self.guard_tags_percentage.append(
                testing_results.guard_tags_with_lora / testing_results.total_behavior_responses * 100
            )

    def print_accuracy_results(self):
        print(f"Average Accuracy with LoRA: {np.mean(self.accuracy_with_lora):.2%}")
        print(f"Average Accuracy without LoRA: {np.mean(self.accuracy_without_lora):.2%}")
        print(f"Average Behavior Adherence with LoRA: {np.mean(self.behavior_accuracy_with_lora):.2%}")
        print(f"Average Behavior Adherence without LoRA: {np.mean(self.behavior_accuracy_without_lora):.2%}")
        print(f"Min Accuracy with LoRA: {np.min(self.accuracy_with_lora):.2%}")
        print(f"Max Accuracy with LoRA: {np.max(self.accuracy_with_lora):.2%}")
        print(f"Min Accuracy without LoRA: {np.min(self.accuracy_without_lora):.2%}")
        print(f"Max Accuracy without LoRA: {np.max(self.accuracy_without_lora):.2%}")
        if self.guard_tags_percentage:
            print(f"\nGuard Tag Statistics:")
            print(f"Average guard tags: {np.mean(self.guard_tags_percentage):.2f}% "
                  f"(±{np.std(self.guard_tags_percentage):.2f}%)")
    
    def write_accuracy_results(self, report_file: TextIO):
        report_file.write(f"\nFinal Statistics:\n")
        report_file.write(f"Average Accuracy with LoRA: {np.mean(self.accuracy_with_lora):.2%}\n")
        report_file.write(f"Average Accuracy without LoRA: {np.mean(self.accuracy_without_lora):.2%}\n")
        report_file.write(f"Average Behavior Adherence with LoRA: {np.mean(self.behavior_accuracy_with_lora):.2%}\n")
        report_file.write(f"Average Behavior Adherence without LoRA: {np.mean(self.behavior_accuracy_without_lora):.2%}\n")
        report_file.write(f"Min Accuracy with LoRA: {np.min(self.accuracy_with_lora):.2%}\n")
        report_file.write(f"Max Accuracy with LoRA: {np.max(self.accuracy_with_lora):.2%}\n")
        report_file.write(f"Min Accuracy without LoRA: {np.min(self.accuracy_without_lora):.2%}\n")
        report_file.write(f"Max Accuracy without LoRA: {np.max(self.accuracy_without_lora):.2%}\n")
        if self.guard_tags_percentage:
            report_file.write(f"\nGuard Tag Statistics:\n")
            report_file.write(f"Average guard tags: {np.mean(self.guard_tags_percentage):.2f}% "
                            f"(±{np.std(self.guard_tags_percentage):.2f}%)\n")