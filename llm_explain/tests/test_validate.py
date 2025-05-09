from llm_explain.llm.validate import validate

if __name__ == "__main__":
    predicate = "is anti-vaccination"
    x_samples = [
        "I am against vaccination because it is dangerous",
        "I am for vaccination because it is good for health",
    ]
    model_name = "gpt-4o"

    for x_sample in x_samples:
        answer = validate(predicate, x_sample, model_name=model_name)
        print(f"{x_sample}: {answer}")


