# Scalable Understanding of Datasets and Models with the Help of Large Language Models

I will make a video tutorial on this topic; stay tuned. This is the library and notebooks to help the audience understand my tutorial. 

### Installation

I recommend creating a conda environment with python >= 3.9 to use this package.

1. Run ```pip install -e .``` to install this package.
2. Set up your openai key in your environment. i.e. ```export OPENAI_API_KEY="[Your OPENAI API KEY]"```

### Usage

This repo supports the bare bone implementation for explaining dataset differences and clusters. 

See the notebooks, ```llm_explain/tests/test_cluster.py```, and ```llm_explain/tests/test_diff.py``` to understand how to use the functions implemented in this repo. 

If you want to build on it, refer to other test files to understand the rest of the repo. 

### Notebooks 

The notebooks illustrate the following sections in the video tutorial.
- 1.1 Core method: the proposer-validator framework
- 1.1 Extension 1: precise explanations.
- 1.1 Extension 2: goal-constrained explanations.
- 1.1 Extension 3: multiple explanations
- 1.2: Explainable clustering

### Related works

```references.pdf``` contains the related works mentioned in our presentation. 

