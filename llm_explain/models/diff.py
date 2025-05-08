from llm_explain.llm.propose import propose_diff
from llm_explain.llm.validate import validate
from openai import OpenAI
from multiprocessing import Pool
import random
import numpy as np
from dataclasses import dataclass

def _propose_round(args: tuple[list[str], list[bool], str | None, str | None, int, int, str, float, OpenAI, bool]) -> list[str]:
    """
    Propose explanations for a round.
    """
    X, Y, context, constraint, proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_detailed = args
    subsampled_x_samples, subsampled_y = balanced_sampling(X, Y, proposer_num_x_samples_per_round)
    return propose_diff(x_samples=subsampled_x_samples, y=subsampled_y, context=context, constraint=constraint, num_explanations=proposer_num_explanations_per_round, model_name=proposer_model_name, temperature=proposer_temperature, client=proposer_client, detailed=proposer_detailed)

def _propose_in_parallel(X: list[str], Y: list[bool], context: str | None, constraint: str | None, proposer_model_name: str, proposer_temperature: float, proposer_client: OpenAI, proposer_detailed: bool, proposer_num_rounds: int, proposer_num_explanations_per_round: int, proposer_num_x_samples_per_round: int, num_processes_max: int) -> list[str]:
    """
    Propose multiple rounds of explanations in parallel.
    """

    all_proposed_explanations: list[str] = []
    with Pool(processes=min(proposer_num_rounds, num_processes_max)) as pool:
        args = [(X, Y, context, constraint, proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_detailed) for _ in range(proposer_num_rounds)]
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


def balanced_sampling(X: list[str], Y: list[bool], num_samples: int) -> tuple[list[str], list[bool]]:
    """
    Balance the sampling of x_samples from negative and positive classes.
    """
    X, Y = np.array(X), np.array(Y, dtype=bool)
    x_samples_positive: list[str] = X[Y]
    x_samples_negative: list[str] = X[~Y]

    random.shuffle(x_samples_positive)
    random.shuffle(x_samples_negative)

    new_x_samples: list[str] = np.concatenate([x_samples_positive[:num_samples], x_samples_negative[:num_samples]])
    new_y: list[bool] = np.concatenate([np.ones(num_samples), np.zeros(num_samples)])
    return new_x_samples, new_y


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
        self.validation_results = [[explanation2x_sample2matches[explanation][x_sample] for x_sample in X] for explanation in all_explanations]

def explain_diff(
        X: list[str], Y: list[bool], 
        context: str | None = None, constraint: str | None = None,
        proposer_model_name: str="gpt-4o", proposer_temperature: float=1.0, proposer_client: OpenAI=None, proposer_detailed: bool=True, proposer_num_rounds: int=12, proposer_num_explanations_per_round: int=5, proposer_num_x_samples_per_round: int=12,
        validator_model_name: str="gpt-4o", validator_client: OpenAI=None, 
        num_processes_max: int=10,
        random_seed: int=42,
) -> ExplainDiffResult:
    random.seed(random_seed)
    X, Y = np.array(X), np.array(Y)

    # propose explanations, get a list of explanations (\phi)
    all_proposed_explanations: list[str] = _propose_in_parallel(X=X, Y=Y, context=context, constraint=constraint, 
                                                     proposer_model_name=proposer_model_name, proposer_temperature=proposer_temperature, proposer_client=proposer_client, proposer_detailed=proposer_detailed, proposer_num_rounds=proposer_num_rounds, proposer_num_explanations_per_round=proposer_num_explanations_per_round, proposer_num_x_samples_per_round=proposer_num_x_samples_per_round, num_processes_max=num_processes_max)
    
    # validate explanations, get a dict of explanation (\phi) -> x_sample (x) -> bool [[\phi]](x)
    explanation2x_sample2matches: dict[str, dict[str, bool]] = _validate_in_parallel(all_proposed_explanations, X, 
                                                       validator_model_name, validator_client, num_processes_max)
    
    # check accuracy of each explanation
    explanation_balanced_accuracy: dict[str, float] = get_balanced_accuracy_of_explanations(explanation2x_sample2matches, X, Y)
    return ExplainDiffResult(X, Y, explanation_balanced_accuracy, all_proposed_explanations, explanation2x_sample2matches)


def KSparseRegression(X: np.ndarray, Y: np.ndarray, K: int) -> np.ndarray:
    """
    Perform K-sparse regression on X and Y.
    X is a 2D numpy array of shape (n_samples, n_features).
    Y is a 1D numpy array of shape (n_samples,).
    K is the number of features to select.

    Returns a 1D numpy array of shape (n_features,) containing the coefficients of the selected features.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty="l1", solver="liblinear")
    model.fit(X, Y)
    top_K_features = np.argsort(np.abs(model.coef_.flatten()))[-K:]

    # select the top K features and run regression again
    X_top_K = X[:, top_K_features]
    model.fit(X_top_K, Y)

    coefs = np.zeros(X.shape[1])
    coefs[top_K_features] = model.coef_
    return coefs