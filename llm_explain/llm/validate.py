from llm_explain.utils import extract_tag_from_output, logger
from openai import OpenAI
from pydantic import BaseModel
class Answer(BaseModel):
    answer: str

ANSWER_TAG: str = "answer"

VALIDATION_EXAMPLES: list[dict] = [
    {
        "predicate": "has a positive sentiment",
        "x_sample": "this movie is bad",
        "answer": "no"
    },
    {
        "predicate": "contains a green object",
        "x_sample": "the frog is climbing a tree",
        "answer": "yes"
    },
]

VALIDATION_EXAMPLE_TEMPLATE: str = """
<predicate>{predicate}</predicate>
<x_sample>{x_sample}</x_sample>
"""
ANSWER_TEMPLATE: str = """<{ANSWER_TAG}>{answer}</{ANSWER_TAG}>
"""

VALIDATION_EXAMPLES_STR: str = ""
for example in VALIDATION_EXAMPLES:
    VALIDATION_EXAMPLES_STR += VALIDATION_EXAMPLE_TEMPLATE.format(predicate=example["predicate"], x_sample=example["x_sample"]) + ANSWER_TEMPLATE.format(answer=example["answer"], ANSWER_TAG=ANSWER_TAG)

PROMPT_BODY_TEMPLATE: str = """
Your job is to validate whether an x_sample surrounded by <x_sample> tags satisfies a predicate surrounds by <predicate> tags. Your output should be a yes or no.

{validation_examples_str}

Now validate the following.
{target_example_str}
Just output the answer surrounded by <answer> tags."""

answer_dict: dict[str, bool] = {
    "yes": 1,
    "no": 0,
}

def create_prompt_body(predicate: str, x_sample: str) -> str:
    """
    Create the prompt body for the validation.

    Args:
        predicate (str): The predicate to validate.
        x_sample (str): The x_sample to validate.

    Returns:
        str: The prompt body.
    """
    target_example_str: str = VALIDATION_EXAMPLE_TEMPLATE.format(predicate=predicate, x_sample=x_sample)
    return PROMPT_BODY_TEMPLATE.format(validation_examples_str=VALIDATION_EXAMPLES_STR, target_example_str=target_example_str)



def validate(predicate: str, x_sample: str, model_name: str = "gpt-4o", client: OpenAI = None, temperature: float = 0.0) -> bool | None:
    """
    Validate whether a x_sample x_sample surrounds by <x_sample> tags satisfies a predicate surrounds by <predicate> tags.

    Args:
        predicate (str): The predicate to validate.
        x_sample (str): The x_sample to validate.
        model_name (str): The name of the model to use.
        client (OpenAI): The client to use.
        temperature (float): The temperature to use.

    Returns:
        bool: Whether the x_sample satisfies the predicate. None if the LLM response is invalid.
    """
    if client is None:
        client = OpenAI()
    prompt_body: str = create_prompt_body(predicate, x_sample)
    logger.debug(f"Prompt body: {prompt_body}")

    messages: list[dict] = [
        {"role": "user", "content": prompt_body}
    ]
    raw_output: str = client.responses.parse(
        model=model_name,
        input=messages,
        text_format=Answer,
        temperature=temperature,
    ).output_parsed.answer

    logger.debug(f"Raw response: {raw_output}")

    answer: str = extract_tag_from_output(raw_output, ANSWER_TAG)
    if answer not in ["yes", "no"]:
        logger.warning(f"Invalid raw response {raw_output}")

    return answer_dict.get(answer, None)
