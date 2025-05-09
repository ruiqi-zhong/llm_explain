import random
import numpy as np
from openai import OpenAI
from llm_explain.llm.propose import propose_in_parallel
from llm_explain.llm.validate import validate_in_parallel
from itertools import combinations
from dataclasses import dataclass
from typing import Union
from llm_explain import logger

def select_indices_based_on_validation_results(validation_results: list[list[bool]], num_clusters: int) -> list[int]:
    """
    Select indices based on validation results. 
    This is a brute force approach to find the best clustering. Do not use this in production.
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
    context: Union[str, None] = None,
    constraint: Union[str, None] = None,
    proposer_model_name: str="gpt-4o", proposer_temperature: float=1.0, proposer_client: OpenAI=None, proposer_precise: bool=True, proposer_num_rounds: int=12, proposer_num_explanations_per_round: int=5, proposer_num_x_samples_per_round: int=12,
    validator_model_name: str="gpt-4o", validator_client: OpenAI=None, 
    num_processes_max: int=10,
    random_seed: int=42,
) -> ClusteringResult:
    """
    Explainable clustering with the help of language models.

    Args:
        X: The dataset to cluster.
        num_clusters: The number of clusters to create.
        context: The context to use for the clustering.
        constraint: The constraint to use for the clustering.
        proposer_model_name: The model to use for the proposer.
        proposer_temperature: The temperature to use for the proposer.
        proposer_client: The client to use for the proposer.
        proposer_precise: Whether to use precise explanations.
        proposer_num_rounds: The number of rounds to use for the proposer.
        proposer_num_explanations_per_round: The number of explanations to propose per round.
        proposer_num_x_samples_per_round: The number of x samples to use per round.
        validator_model_name: The model to use for the validator.
        validator_client: The client to use for the validator.
        num_processes_max: The maximum number of processes to use.
        random_seed: The random seed to use.

    Returns:
        A ClusteringResult object.
    """
    random.seed(random_seed)

    logger.info(f"Proposing explanations...")
    # Propose explanations
    all_proposed_explanations: list[str] = propose_in_parallel(X=X, Y=None, context=context, constraint=constraint, proposer_model_name=proposer_model_name, proposer_temperature=proposer_temperature, proposer_client=proposer_client, proposer_precise=proposer_precise, proposer_num_rounds=proposer_num_rounds, proposer_num_explanations_per_round=proposer_num_explanations_per_round, proposer_num_x_samples_per_round=proposer_num_x_samples_per_round, num_processes_max=num_processes_max, task_name="cluster")

    logger.info(f"Validating explanations, {len(all_proposed_explanations)} explanations x {len(X)} x_samples")
    # Validate explanations
    explanation2x_sample2matches: dict[str, dict[str, bool]] = validate_in_parallel(all_proposed_explanations, X, 
                                                       validator_model_name, validator_client, num_processes_max)
    validation_results: list[list[bool]] = [[explanation2x_sample2matches[explanation][x_sample] for x_sample in X] for explanation in all_proposed_explanations]

    logger.info(f"Selecting {num_clusters} clusters from {len(all_proposed_explanations)} candidates")
    # Select the clusters
    selected_indices: list[int] = select_indices_based_on_validation_results(validation_results, num_clusters)

    # aggregate the results
    result: ClusteringResult = ClusteringResult(X, all_proposed_explanations, explanation2x_sample2matches, selected_indices)

    return result