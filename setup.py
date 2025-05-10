from setuptools import setup, find_packages

setup(
    name="llm_explain",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pydantic",
        "requests",
        "numpy",
        "scikit-learn",
        "tqdm"
    ],
    author="Ruiqi Zhong",
    author_email="ruiqizhong1997@gmail.com",
    description="A package for using LLMs to explain datasets",
    license="MIT",
    keywords="llm, explain, dataset, explanation",
    url="https://github.com/ruiqi-zhong/llm_explain",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)