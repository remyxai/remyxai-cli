from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="remyxai",
    version="0.1.8",
    packages=find_packages(include=["remyxai", "remyxai.*"]),
    install_requires=[
        "numpy",
        "onnx",
        "onnxruntime",
        "pillow",
        "requests",
        "tqdm",
        "tritonclient",
        "huggingface_hub",
        "datasets",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "remyxai=remyxai.cli.commands:cli",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
