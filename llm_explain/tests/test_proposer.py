from llm_explain.llm.propose import propose

if __name__ == "__main__":
    samples_A = ["cat", "dog", "fish"]
    samples_B = ["carrot", "potato", "apple"]

    detailed_explanations = propose(samples_A, samples_B, num_explanations=3, model_name="gpt-4o", detailed=True)
    simple_explanations = propose(samples_A, samples_B, num_explanations=3, model_name="gpt-4o", detailed=False)

    print("Here are the detailed explanations:")
    for explanation in detailed_explanations:
        print("-", explanation)

    print("Here are the simple explanations:")
    for explanation in simple_explanations:
        print("-", explanation)