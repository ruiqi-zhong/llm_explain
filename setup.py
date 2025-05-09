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
        "scikit-learn"
    ],
)