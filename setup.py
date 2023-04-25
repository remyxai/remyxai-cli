from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="remyxai",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["requests"],
    entry_points={
        "console_scripts": [
            "remyxai=remyxai.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
