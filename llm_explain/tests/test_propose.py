from llm_explain.llm.propose import propose_diff

if __name__ == "__main__":
    x_samples = ["cat", "dog", "fish", "carrot", "potato", "apple"]
    y = [False, False, False, True, True, True]

    detailed_explanations = propose_diff(x_samples, y, num_explanations=3, model_name="gpt-4o", detailed=True)
    simple_explanations = propose_diff(x_samples, y, num_explanations=3, model_name="gpt-4o", detailed=False)

    print("Here are the detailed explanations:")
    for explanation in detailed_explanations:
        print("-", explanation)

    print("Here are the simple explanations:")
    for explanation in simple_explanations:
        print("-", explanation)