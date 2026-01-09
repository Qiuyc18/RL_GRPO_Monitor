from setuptools import find_packages, setup

setup(
    name="rl-llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gradio",
        "pandas",
        "plotly",
        "transformers",
        "accelerate",
        "huggingface_hub",
    ],
)
