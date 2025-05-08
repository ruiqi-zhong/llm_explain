import random
import numpy as np
from openai import OpenAI
from llm_explain.llm.propose import propose_in_parallel
from llm_explain.llm.validate import validate_in_parallel
from itertools import combinations
from dataclasses import dataclass

def select_indices_based_on_validation_results(validation_results: list[list[bool]], num_clusters: int) -> list[int]:
    """
    Select indices based on validation results.
    """
    validation_results = np.array(validation_results)

    num_candidate_clusters, num_samples = validation_results.shape

    minimal_cost, minimal_indices = float("inf"), None

    # brute force enumerate all possible combinations of indices
    target = np.ones(num_samples)
    for indices in combinations(list(range(num_candidate_clusters)), num_clusters):
        indices = list(indices)
        coverage = np.sum(validation_results[indices], axis=0)
        cost = np.sum(np.abs(coverage - target))
        if cost < minimal_cost:
            minimal_cost = cost
            minimal_indices = indices

    return minimal_indices

@dataclass
class ClusteringResult:
    X: list[str]
    all_proposed_explanations: list[str]
    explanation2x_sample2matches: dict[str, dict[str, bool]]
    selected_indices: list[int]
    selected_explanations: list[str]
    selected_clusters: list[list[str]]

    def __init__(self, X: list[str], all_proposed_explanations: list[str], explanation2x_sample2matches: dict[str, dict[str, bool]], selected_indices: list[int]):
        self.X = X
        self.all_proposed_explanations = all_proposed_explanations
        self.validation_results = np.array([[explanation2x_sample2matches[explanation][x_sample] for x_sample in X] for explanation in all_proposed_explanations])
        self.selected_indices = selected_indices
        self.selected_explanations = [all_proposed_explanations[i] for i in selected_indices]
        self.selected_clusters = [
            [X[i] for i in range(len(X)) if self.validation_results[j, i]]
            for j in self.selected_indices
        ]

    def __repr__(self) -> str:
        output: str = ""
        for i, (explanation, cluster) in enumerate(zip(self.selected_explanations, self.selected_clusters)):
            output += f"Cluster {i+1}: {explanation}\n"
            for x in cluster[:5]:
                output += f"  - {x}\n"
            output += "\n"
        return output



def explain_cluster(
    X: list[str],
    num_clusters,
    context: str | None = None,
    constraint: str | None = None,
    proposer_model_name: str="gpt-4o", proposer_temperature: float=1.0, proposer_client: OpenAI=None, proposer_detailed: bool=True, proposer_num_rounds: int=12, proposer_num_explanations_per_round: int=5, proposer_num_x_samples_per_round: int=12,
    validator_model_name: str="gpt-4o", validator_client: OpenAI=None, 
    num_processes_max: int=10,
    random_seed: int=42,
):
    random.seed(random_seed)

    all_proposed_explanations: list[str] = propose_in_parallel(X=X, Y=None, context=context, constraint=constraint, proposer_model_name=proposer_model_name, proposer_temperature=proposer_temperature, proposer_client=proposer_client, proposer_detailed=proposer_detailed, proposer_num_rounds=proposer_num_rounds, proposer_num_explanations_per_round=proposer_num_explanations_per_round, proposer_num_x_samples_per_round=proposer_num_x_samples_per_round, num_processes_max=num_processes_max, task_name="cluster")

    explanation2x_sample2matches: dict[str, dict[str, bool]] = validate_in_parallel(all_proposed_explanations, X, 
                                                       validator_model_name, validator_client, num_processes_max)
    validation_results: list[list[bool]] = [[explanation2x_sample2matches[explanation][x_sample] for x_sample in X] for explanation in all_proposed_explanations]
    selected_indices: list[int] = select_indices_based_on_validation_results(validation_results, num_clusters)
    result: ClusteringResult = ClusteringResult(X, all_proposed_explanations, explanation2x_sample2matches, selected_indices)

    return result