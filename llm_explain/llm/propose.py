from pydantic import BaseModel
from llm_explain.utils import extract_tag_from_output
from llm_explain import logger
from openai import OpenAI
import numpy as np
from multiprocessing import Pool
import random
from typing import Union
import tqdm

PREDICATE_TAG = "predicate"
precise_predicates: list[str] = [
    "uses double negation; specifically, there is a sentence in the text that uses negation twice. For example, 'the pilot did not never had an accident'",
    "has a conservative stance; specifically, the overall text exhibits a conservative political stance (e.g. pro-life, deregulation, etc). For example, 'Fetus is sentient so we should not abort.'"
]

precise_format_description: str = "about a text followed by an explanation and an example that satisfies the predicate."

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


def _prepare_prefix_for_proposer_prompt(precise: bool):
    format_description: str = precise_format_description if precise else simple_format_description
    example_predicates: list[str] = precise_predicates if precise else simple_predicates

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


def create_proposer_diff_prompt_body(x_samples: list[str], y: list[bool], num_explanations: int, context: str = None, constraint: str = None, precise: bool = False) -> str:
    """
    Create the prompt body for the proposer. It will be used as part of the user message.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        precise (bool): Whether to use precise or simple descriptions.

    Returns:
        str: The prompt body.
    """

    x_samples, y = np.array(x_samples), np.array(y, dtype=bool)

    negative_block: str = create_block_of_x_samples(x_samples[~y], "Negative class sample")
    positive_block: str = create_block_of_x_samples(x_samples[y], "Positive class sample")

    format_description, example_predicates_str = _prepare_prefix_for_proposer_prompt(precise)
    suffix: str = _prepare_suffix_for_proposer_prompt(context, constraint)
    prompt_body: str = PROPOSER_DIFF_PROMPT_BODY_TEMPLATE.format(negative_block=negative_block, positive_block=positive_block, num_explanations=num_explanations, format_description=format_description, example_predicates_str=example_predicates_str, PREDICATE_TAG=PREDICATE_TAG, goal_constrained_suffix=suffix)
    return prompt_body

def create_proposer_cluster_prompt_body(x_samples: list[str], num_explanations: int, context: str = None, constraint: str = None, precise: bool = False) -> str:
    """
    Create the prompt body for the proposer. It will be used as part of the user message.

    Args:
        x_samples (list[str]): The list of x_samples.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        precise (bool): Whether to use precise or simple descriptions.

    Returns:
        str: The prompt body.
    """

    block: str = create_block_of_x_samples(x_samples, "Sample")

    format_description, example_predicates_str = _prepare_prefix_for_proposer_prompt(precise)
    suffix: str = _prepare_suffix_for_proposer_prompt(context, constraint)
    prompt_body: str = PROPOSER_CLUSTER_PROMPT_BODY_TEMPLATE.format(block=block, num_explanations=num_explanations, format_description=format_description, example_predicates_str=example_predicates_str, PREDICATE_TAG=PREDICATE_TAG, goal_constrained_suffix=suffix)
    return prompt_body

def get_proposer_messages(x_samples: list[str], y: Union[list[bool], None] = None, num_explanations: int = 5, context: str = None, constraint: str = None, precise: bool = True, task_name: str = "diff") -> list[dict]:
    """
    Get the messages for the proposer.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        precise (bool): Whether to use precise or simple descriptions.

    Returns:
        list[dict]: The messages for the proposer.
    """
    if task_name == "diff":
        prompt_body: str = create_proposer_diff_prompt_body(x_samples=x_samples, y=y, num_explanations=num_explanations, context=context, constraint=constraint, precise=precise)
    elif task_name == "cluster":
        prompt_body: str = create_proposer_cluster_prompt_body(x_samples=x_samples, num_explanations=num_explanations, context=context, constraint=constraint, precise=precise)
    logger.debug(f"Prompt body: {prompt_body}")

    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant that analyzes text x_samples."},
        {"role": "user", "content": prompt_body}
    ]

    return messages

def postprocess_explanations_list_from_raw_output(raw_output: ExplanationList, num_explanations: int) -> list[str]:
    """
    Postprocess the explanations list from the raw output.
    """
    explanations_list: list[Union[str, None]] = [extract_tag_from_output(explanation, PREDICATE_TAG) for explanation in raw_output.explanations]
    logger.debug(f"Explanations list unfiltered: {explanations_list}")
    filtered_explanations_list: list[str] = [explanation for explanation in explanations_list if explanation is not None]
    if len(filtered_explanations_list) != num_explanations:
        logger.warning(f"Proposer expected {num_explanations} explanations, but got {len(filtered_explanations_list)}")
    
    return filtered_explanations_list

def propose(x_samples: list[str], y: Union[list[bool], None] = None, num_explanations: int = 5, context: str = None, constraint: str = None, model_name: str = "gpt-4o", precise: bool = True, client: OpenAI = None, temperature: float = 1.0, task_name: str = "diff") -> list[str]:
    """
    Propose a list of descriptions for the x_samples in the positive class.

    Args:
        x_samples (list[str]): The list of x_samples.
        y (list[bool]): The list of labels.
        num_explanations (int): The number of descriptions to propose.
        context (str): The context to use for the descriptions.
        constraint (str): The constraint to use for the descriptions.
        model_name (str): The name of the model to use.
        precise (bool): Whether to use precise or simple descriptions.
        client (OpenAI): The client to use for the descriptions.
        temperature (float): The temperature to use for the descriptions.

    Returns:
        list[str]: The list of descriptions.
    """
    if client is None:
        client = OpenAI()
    x_samples = np.array(x_samples)

    if y is not None and task_name == "diff":
        y = np.array(y, dtype=bool)

    # prepare the messages
    if task_name == "diff":
        messages: list[dict] = get_proposer_messages(x_samples=x_samples, y=y, num_explanations=num_explanations, context=context, constraint=constraint, precise=precise, task_name=task_name)
    elif task_name == "cluster":
        messages: list[dict] = get_proposer_messages(x_samples=x_samples, num_explanations=num_explanations, context=context, constraint=constraint, precise=precise, task_name=task_name)
    else:
        raise ValueError(f"Invalid task name: {task_name}")

    # send the message to the model and parse the output
    raw_output: list[str] = client.responses.parse(
        model=model_name,
        input=messages,
        text_format=ExplanationList,
        temperature=temperature,
    ).output_parsed
    logger.debug(f"Raw response: {raw_output}")

    return postprocess_explanations_list_from_raw_output(raw_output, num_explanations)


def balanced_sampling(X: list[str], Y: list[bool], num_samples: int) -> tuple[list[str], list[bool]]:
    """
    Randomly sample num_samples from positive and negative classes each in case the two classes are imbalanced.
    """
    X, Y = np.array(X), np.array(Y, dtype=bool)
    x_samples_positive: list[str] = X[Y][:num_samples]
    x_samples_negative: list[str] = X[~Y][:num_samples]

    random.shuffle(x_samples_positive)
    random.shuffle(x_samples_negative)

    new_x_samples: list[str] = np.concatenate([x_samples_positive, x_samples_negative])
    new_y: list[bool] = np.concatenate([np.ones(len(x_samples_positive)), np.zeros(len(x_samples_negative))])
    return new_x_samples, new_y


def _propose_round(args: tuple[list[str], list[bool], Union[str, None], Union[str, None], int, int, str, float, OpenAI, bool, str]) -> list[str]:
    """
    Propose explanations for a round, used mostly as a helper function for parallelization.
    """
    X, Y, context, constraint, proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_precise, task_name = args
    if task_name == "diff":
        subsampled_x_samples, subsampled_y = balanced_sampling(X, Y, proposer_num_x_samples_per_round)
    elif task_name == "cluster":
        assert Y is None
        subsampled_x_samples, subsampled_y = random.sample(X, proposer_num_x_samples_per_round), None
    return propose(x_samples=subsampled_x_samples, y=subsampled_y, context=context, constraint=constraint, num_explanations=proposer_num_explanations_per_round, model_name=proposer_model_name, temperature=proposer_temperature, client=proposer_client, precise=proposer_precise, task_name=task_name)

def propose_in_parallel(X: list[str], Y: list[bool], context: Union[str, None], constraint: Union[str, None], proposer_model_name: str, proposer_temperature: float, proposer_client: OpenAI, proposer_precise: bool, proposer_num_rounds: int, proposer_num_explanations_per_round: int, proposer_num_x_samples_per_round: int, num_processes_max: int, task_name: str) -> list[str]:
    """
    Propose multiple rounds of explanations in parallel. 
    """
    all_proposed_explanations: list[str] = []

    with Pool(processes=min(proposer_num_rounds, num_processes_max)) as pool:
        args = [(X, Y, context, constraint, proposer_num_x_samples_per_round, proposer_num_explanations_per_round, proposer_model_name, proposer_temperature, proposer_client, proposer_precise, task_name) for _ in range(proposer_num_rounds)]
        results = list(tqdm.tqdm(pool.imap(_propose_round, args), total=proposer_num_rounds, desc="Proposing explanations"))
        for proposed_explanations in results:
            all_proposed_explanations.extend(proposed_explanations)
    return all_proposed_explanations