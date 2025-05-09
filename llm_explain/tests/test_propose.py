from llm_explain.llm.propose import propose

if __name__ == "__main__":
    x_samples = ["cat", "dog", "fish", "carrot", "potato", "apple"]
    y = [False, False, False, True, True, True]

    precise_explanations = propose(x_samples=x_samples, y=y, num_explanations=3, model_name="gpt-4o", precise=True, task_name="diff")
    simple_explanations = propose(x_samples=x_samples, y=y, num_explanations=3, model_name="gpt-4o", precise=False, task_name="diff")

    print("Here are the precise explanations:")
    for explanation in precise_explanations:
        print("-", explanation)

    print("Here are the simple explanations:")
    for explanation in simple_explanations:
        print("-", explanation)

    cluster_explanations = propose(x_samples=x_samples, num_explanations=2, model_name="gpt-4o", precise=True, task_name="cluster")
    print("Here are the cluster explanations:")
    for explanation in cluster_explanations:
        print("-", explanation)
