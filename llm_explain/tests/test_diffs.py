from llm_explain.models.diff import explain_diff, ExplainDiffResult, KSparseRegression
from data.samples import get_goal_driven_examples
import numpy as np

debug_case_no_detail = {
    "X": ["cat", "dog", "fish", "carrot", "potato", "apple"],
    "Y": [False, False, False, True, True, True],
    "proposer_num_rounds": 2,
    "proposer_num_explanations_per_round": 3,
    "proposer_detailed": False,
}

debug_case_with_detail = {
    "X": ["cat", "dog", "fish", "carrot", "potato", "apple"],
    "Y": [False, False, False, True, True, True],
    "proposer_num_rounds": 2,
    "proposer_num_explanations_per_round": 3,
    "proposer_detailed": True,
}

airline_goal_driven_toy_example = {
    **get_goal_driven_examples(seed=42),
    "proposer_num_rounds": 5,
    "proposer_num_explanations_per_round": 3,
    "proposer_detailed": True,
}

airline_goal_driven_toy_example_no_constraint = {
    **get_goal_driven_examples(seed=42, with_constraint=False),
    "proposer_num_rounds": 5,
    "proposer_num_explanations_per_round": 3,
    "proposer_detailed": True,
}

def test_KSparseRegression():
    X = np.random.randn(100, 10)
    K = 3
    coefs = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    logits = X @ coefs
    Y = np.random.binomial(1, 1 / (1 + np.exp(-logits)))

    learned_coefs = KSparseRegression(X, Y, K)
    print("learned coefs: ", learned_coefs)
    print("true coefs: ", coefs)


if __name__ == "__main__":
    # run_case = airline_goal_driven_toy_example_no_constraint
    run_case = airline_goal_driven_toy_example
    result: ExplainDiffResult = explain_diff(**run_case)

    print_top_n = 3
    # Get indices of top 5 scoring explanations
    top_n_indices = sorted(range(len(result.explanation_balanced_accuracy)), 
                         key=lambda i: result.explanation_balanced_accuracy[i],
                         reverse=True)[:print_top_n]
    
    for i, explanation_idx in enumerate(top_n_indices):
        print(f"{explanation_idx}: {result.all_explanations[explanation_idx]}")
        print(f"  accuracy: {result.explanation_balanced_accuracy[explanation_idx]}")
        print(f"  validation results: {result.validation_results[explanation_idx]}")

    feature_vals = np.array(result.validation_results).T

    explanation_coefs = KSparseRegression(feature_vals, result.Y, K=2)

    for explanation_idx, explanation in enumerate(result.all_explanations):
        if explanation_coefs[explanation_idx] != 0:
            print(f"{explanation_idx}: {explanation}")
            print(f"  coef: {explanation_coefs[explanation_idx]}")
