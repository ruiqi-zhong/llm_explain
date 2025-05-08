from llm_explain.models.diff import explain_diff, ExplainDiffResult

X = ["cat", "dog", "fish", "carrot", "potato", "apple"]
Y = [False, False, False, True, True, True]
result: ExplainDiffResult = explain_diff(X, Y, proposer_num_rounds=2, proposer_num_explanations_per_round=3)

for explanation_idx, explanation in enumerate(result.all_explanations):
    print(f"{explanation_idx}: {explanation}")
    print(f"  accuracy: {result.explanation_balanced_accuracy[explanation_idx]}")
    print(f"  validation results: {result.explanation2x_sample2matches[explanation_idx]}")
