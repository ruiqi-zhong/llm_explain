# Scalable Understanding of Datasets and Models with the Help of Large Language Models

I made a video tutorial on this topic [here](https://youtu.be/TEwUkr9n1s8). 

I wrote a thesis on this research direction, and you can see it [here](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-78.pdf). 

This is the library and notebooks to help the audience understand the section 1 of my tutorial.

## Installation

I recommend creating a conda environment with python >= 3.9 to use this package.

1. Set up your openai key in your environment. i.e. ```export OPENAI_API_KEY="[Your OPENAI API KEY]"```
2. Installation

**Option 1**: clone and install locally (for developers)
- ```git clone git@github.com:ruiqi-zhong/llm_explain.git```
- ```cd llm_explain```
- ```pip install -e .```

**Option 2**: install from github repo

```pip3 install --no-cache-dir -v git+https://github.com/ruiqi-zhong/llm_explain.git```

**Option 3**: install from pypi

```pip3 install llm-explain```

## Usage

This repo supports the bare bone implementation for explaining dataset differences and clusters. 

See the notebooks, ```llm_explain/tests/test_cluster.py```, and ```llm_explain/tests/test_diff.py``` to understand how to use the functions implemented in this repo. 

If you want to build on it, refer to other test files to understand the rest of the repo. 

## A quick example after installation

run python: 
```
>>> from llm_explain.models.diff import explain_diff                                                                                                                           
>>> explain_diff(["cat", "dog", "fish", "carrot", "potato", "apple"], [False, False, False, True, True, True], proposer_num_rounds=2, proposer_num_explanations_per_round=2)
```

You will get outputs similar to the following in fewer than 30 seconds:

```
Printing top 3 explanations:
Explanation: refers to a plant-based item; specifically, the text mentions items that grow from plants, including vegetables and fruits. For example, 'carrot' is a type of root vegetable.
Accuracy: 1.0

Explanation: is a type of food; specifically, the text refers to items commonly recognized as food, typically vegetables or fruits. For example, 'apple' is known to be a fruit consumed as food.
Accuracy: 0.8333333333333333

Explanation: mentions a type of food; specifically, the text refers to something that is commonly eaten by humans. For example, 'This apple is very juicy.'
Accuracy: 0.8333333333333333
```

## Notebooks 

The notebooks illustrate the following sections in the video tutorial.
- 1.1 Core method: the proposer-validator framework
- 1.1 Extension 1: precise explanations.
- 1.1 Extension 2: goal-constrained explanations.
- 1.1 Extension 3: multiple explanations
- 1.2: Explainable clustering

## Related works

```related/references.pdf``` contains the related works mentioned in our presentation. ```related/main.tex``` contains the latex source file and ```references.bib``` contains the bibtex citations.

