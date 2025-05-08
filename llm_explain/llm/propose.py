from pydantic import BaseModel
from llm_explain.utils import extract_tag_from_output, logger
from openai import OpenAI
import numpy as np

PREDICATE_TAG = "predicate"
detailed_predicates: list[str] = [
    "uses double negation; specifically, there is a sentence in the text that uses negation twice. For example, 'the pilot did not never had an accident'",
    "has a conservative stance; specifically, the overall text exhibits a conservative political stance (e.g. pro-life, deregulation, etc). For example, 'Fetus is sentient so we should not abort.'"
]

detailed_format_description: str = "about a text followed by an explanation and an example that satisfies the predicate."

simple_predicates: list[str] = [
    "uses double negation",
    "has a conservative stance"
]

simple_format_description: str = "about a text."

PROPOSER_DIFF_PROMPT_BODY_TEMPLATE: str = """
Here are two sets of text x_samples.

Some x_samples from the negative class:
{negative_block}

Some x_samples from the positive class:
{positive_block}

We want to understand what kind of text x_samples are more likely to be in the positive class. Please suggest me at most {num_explanations} descriptions. Each of them needs to be a predicate {format_description}. Each predicate needs to be wrapped in <{PREDICATE_TAG}> tags.

Here are some example predicates:
{example_predicates_str}

{goal_constrained_suffix}
"""

PROPOSER_CLUSTER_PROMPT_BODY_TEMPLATE: str = """
Here are some x_samples.

{block}

We want to cluster the x_samples. Please suggest me at most {num_explanations} descriptions. Each of them needs to be a predicate {format_description}. Each predicate needs to be wrapped in <{PREDICATE_TAG}> tags.

Here are some example predicates:
{example_predicates_str}

{goal_constrained_suffix}
"""


class ExplanationList(BaseModel):
    explanations: list[str]


def create_block_of_x_samples(x_samples: list[str], prefix: str) -> str:
    """
    Given a list of x_samples, concatenate them with a prefix and a number.
    e.g. Sample A.1: "Hello, world!"

    Args:
        x_samples (list[str]): The list of x_samples to concatenate.
        prefix (str): The prefix to prepend to each sample.

    Returns:
        str: The concatenated x_samples.
    """

    return "\n".join([f"{prefix}.{i}: {sample}" for i, sample in enumerate(x_samples)])


def _prepare_prefix_for_proposer_prompt(detailed: bool):
    format_description: str = detailed_format_description if detailed else simple_format_description
    example_predicates: list[str] = detailed_predicates if detailed else simple_predicates

    example_predicates_str: str = "\n".join([f"<{PREDICATE_TAG}>{predicate}</{PREDICATE_TAG}>" for predicate in example_predicates])

    return format_description, example_predicates_str

def _prepare_suffix_for_proposer_prompt(context: str = None, constraint: str = None):
    suffix: str = ""
    if context:
        suffix += f"Here is some context about the text x_samples: {context}.\n"

    if constraint:
        suffix += f"You need to follow the constraint: {constraint}.\n"
    
    suffix += f"Just output the predicates surrounded by <{PREDICATE_TAG}> tags. Do not include any other text."
    return suffix


def create_proposer_diff_prompt_body(x_samples: list[str], y: list[bool], num_explanations: int, context: str = None, constraint: str = None, detailed: bool = False) -> str:
    """
    Create the prompt body for the proposer. It will be used as part of the user message.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        detailed (bool): Whether to use detailed or simple descriptions.

    Returns:
        str: The prompt body.
    """

    negative_block: str = create_block_of_x_samples(x_samples[~y], "Negative class sample")
    positive_block: str = create_block_of_x_samples(x_samples[y], "Positive class sample")

    format_description, example_predicates_str = _prepare_prefix_for_proposer_prompt(detailed)
    suffix: str = _prepare_suffix_for_proposer_prompt(context, constraint)
    prompt_body: str = PROPOSER_DIFF_PROMPT_BODY_TEMPLATE.format(negative_block=negative_block, positive_block=positive_block, num_explanations=num_explanations, format_description=format_description, example_predicates_str=example_predicates_str, PREDICATE_TAG=PREDICATE_TAG, goal_constrained_suffix=suffix)
    return prompt_body


def get_proposer_messages(x_samples: list[str], y: list[bool], num_explanations: int, context: str = None, constraint: str = None, detailed: bool = True) -> list[dict]:
    """
    Get the messages for the proposer.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        detailed (bool): Whether to use detailed or simple descriptions.

    Returns:
        list[dict]: The messages for the proposer.
    """
    
    prompt_body: str = create_proposer_diff_prompt_body(x_samples, y, num_explanations, context, constraint, detailed=detailed)
    logger.debug(f"Prompt body: {prompt_body}")

    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant that analyzes text x_samples."},
        {"role": "user", "content": prompt_body}
    ]

    return messages

def propose_diff(x_samples: list[str], y: list[bool] | None = None, num_explanations: int = 5, context: str = None, constraint: str = None, model_name: str = "gpt-4o", detailed: bool = True, client: OpenAI = None, temperature: float = 1.0) -> list[str]:
    """
    Propose a list of descriptions for the x_samples in the positive class.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        model_name (str): The name of the model to use.
        detailed (bool): Whether to use detailed or simple descriptions.
        client (OpenAI): The client to use for the descriptions.
        temperature (float): The temperature to use for the descriptions.

    Returns:
        list[str]: The list of descriptions.
    """
    if client is None:
        client = OpenAI()
    x_samples = np.array(x_samples)
    if y is not None:
        y = np.array(y, dtype=bool)

    # prepare the messages
    messages: list[dict] = get_proposer_messages(x_samples, y, num_explanations, context, constraint, detailed)

    # send the message to the model and parse the output
    raw_output: list[str] = client.responses.parse(
        model=model_name,
        input=messages,
        text_format=ExplanationList,
        temperature=temperature,
    ).output_parsed.explanations
    logger.debug(f"Raw response: {raw_output}")

    # parse the explanations from the output, filter out the None values, and check if the number of explanations is correct
    explanations_list: list[str | None] = [extract_tag_from_output(explanation, PREDICATE_TAG) for explanation in raw_output]
    logger.debug(f"Explanations list unfiltered: {explanations_list}")
    filtered_explanations_list: list[str] = [explanation for explanation in explanations_list if explanation is not None]
    if len(filtered_explanations_list) != num_explanations:
        logger.warning(f"Proposer expected {num_explanations} explanations, but got {len(filtered_explanations_list)}")
    
    return filtered_explanations_list