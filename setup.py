from setuptools import find_packages, setup

setup(
    name="rl-grpo-monitor",
    version="0.1.0",
    packages=find_packages(include=["monitor", "monitor.*"]),
    install_requires=[
        "numpy",
        "gradio",
        "pandas",
        "plotly",
        "python-dotenv",
    ],
    extras_require={
        "nvidia": ["nvidia-ml-py"],
        "amd": ["amdsmi"],
        "data": ["datasets", "huggingface_hub>=0.34.0,<1.0"],
    },
)
