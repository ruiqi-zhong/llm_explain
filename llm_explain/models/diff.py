from llm_explain.llm.propose import propose
from llm_explain.llm.validate import validate
from openai import OpenAI
from multiprocessing import Pool
import random
import numpy as np
from dataclasses import dataclass

def _propose_round(args: tuple[list[str], list[str], int, int, str, float, OpenAI, bool]) -> list[str]:
    """
    Propose explanations for a round.
    """
    x_samples_A, x_samples_B, proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_detailed = args
    random.shuffle(x_samples_A)
    random.shuffle(x_samples_B)
    x_subsample_A = x_samples_A[:proposer_num_x_samples_per_round]
    x_subsample_B = x_samples_B[:proposer_num_x_samples_per_round]
    return propose(x_samples_A=x_subsample_A, x_samples_B=x_subsample_B, num_explanations=proposer_num_explanations_per_round, model_name=proposer_model_name, temperature=proposer_temperature, client=proposer_client, detailed=proposer_detailed)

def _propose_in_parallel(X: list[str], Y: list[bool], proposer_model_name: str, proposer_temperature: float, proposer_client: OpenAI, proposer_detailed: bool, proposer_num_rounds: int, proposer_num_explanations_per_round: int, proposer_num_x_samples_per_round: int, num_processes_max: int) -> list[str]:
    """
    Propose multiple rounds of explanations in parallel.
    """
    x_samples_A: list[str] = X[~Y]
    x_samples_B: list[str] = X[Y]
    all_proposed_explanations: list[str] = []
    with Pool(processes=min(proposer_num_rounds, num_processes_max)) as pool:
        args = [(x_samples_A.copy(), x_samples_B.copy(), proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_detailed) for _ in range(proposer_num_rounds)]
        results = pool.map(_propose_round, args)
        for proposed_explanations in results:
            all_proposed_explanations.extend(proposed_explanations)
    return all_proposed_explanations


def _validate_round(args: tuple[str, str, str, OpenAI]) -> bool:
    """
    Validate an explanation for a round.
    """
    explanation, x_sample, validator_model_name, validator_client = args
    return validate(predicate=explanation, x_sample=x_sample, model_name=validator_model_name, client=validator_client)

def _validate_in_parallel(explanations: list[str], X: list[str], validator_model_name: str, validator_client: OpenAI, num_processes_max: int) -> dict[str, dict[str, bool]]:
    """
    Validate multiple explanations on multiple x_samples in parallel.
    """
    validation_tasks = []
    for explanation in explanations:
        for x_sample in X:
            validation_tasks.append((explanation, x_sample, validator_model_name, validator_client))

    with Pool(processes=min(len(validation_tasks), num_processes_max)) as pool:
        validation_results = pool.map(_validate_round, validation_tasks)

    explanation2x_sample2matches = {}
    for (explanation, x_sample, _, _), result in zip(validation_tasks, validation_results):
        if explanation not in explanation2x_sample2matches:
            explanation2x_sample2matches[explanation] = {}
        explanation2x_sample2matches[explanation][x_sample] = result

    return explanation2x_sample2matches

def get_balanced_accuracy(preds: np.ndarray, Y: np.ndarray) -> float:
    """
    Get the balanced accuracy of a set of predictions; average of the accuracy on positive and negative classes.
    """
    acc_on_pos = np.mean(preds[Y] == 1)
    acc_on_neg = np.mean(preds[~Y] == 0)
    return (acc_on_pos + acc_on_neg) / 2

def get_balanced_accuracy_of_explanations(explanation2x_sample2matches: dict[str, dict[str, bool]], X: list[str], Y: list[bool]) -> dict[str, float]:
    """
    Get the balanced accuracy of a set of explanations on a set of x_samples.
    """
    explanation_accuracy = {}
    for explanation in explanation2x_sample2matches:
        pred_of_explanation = np.array([explanation2x_sample2matches[explanation][x_sample] for x_sample in X])
        explanation_accuracy[explanation] = get_balanced_accuracy(pred_of_explanation, Y)
    return explanation_accuracy

@dataclass
class ExplainDiffResult:
    X: list[str]
    Y: list[bool]
    explanation_balanced_accuracy: dict[str, float]
    all_explanations: list[str]
    validation_results: list[list[bool]]

    def __init__(self, X: list[str], Y: list[bool], explanation_balanced_accuracy: dict[str, float], all_explanations: list[str], explanation2x_sample2matches: dict[str, dict[str, bool]]):
        self.X = X
        self.Y = Y
        self.explanation_balanced_accuracy = [explanation_balanced_accuracy[explanation] for explanation in all_explanations]
        self.all_explanations = all_explanations
        self.explanation2x_sample2matches = [[explanation2x_sample2matches[explanation][x_sample] for x_sample in X] for explanation in all_explanations]

def explain_diff(
        X: list[str], Y: list[bool], 
        proposer_model_name: str="gpt-4o", proposer_temperature: float=1.0, proposer_client: OpenAI=None, proposer_detailed: bool=True, proposer_num_rounds: int=12, proposer_num_explanations_per_round: int=5, proposer_num_x_samples_per_round: int=12,
        validator_model_name: str="gpt-4o", validator_client: OpenAI=None, 
        num_processes_max: int=10,
        random_seed: int=42,
):
    random.seed(random_seed)
    X, Y = np.array(X), np.array(Y)

    # propose explanations, get a list of explanations (\phi)
    all_proposed_explanations: list[str] = _propose_in_parallel(X, Y, 
                                                     proposer_model_name, proposer_temperature, proposer_client, proposer_detailed, proposer_num_rounds, proposer_num_explanations_per_round, proposer_num_x_samples_per_round, num_processes_max)
    
    # validate explanations, get a dict of explanation (\phi) -> x_sample (x) -> bool [[\phi]](x)
    explanation2x_sample2matches: dict[str, dict[str, bool]] = _validate_in_parallel(all_proposed_explanations, X, 
                                                       validator_model_name, validator_client, num_processes_max)
    
    # check accuracy of each explanation
    explanation_balanced_accuracy: dict[str, float] = get_balanced_accuracy_of_explanations(explanation2x_sample2matches, X, Y)
    return ExplainDiffResult(X, Y, explanation_balanced_accuracy, all_proposed_explanations, explanation2x_sample2matches)