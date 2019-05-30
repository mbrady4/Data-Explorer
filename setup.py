""" Data Explorer - A collection of functions to accelerate exploration of a Dataset
"""

import setuptools

REQUIRED = [
    "numpy",
    "pandas"
]

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="data-explorer",
    version="0.0.6",
    author="Michael W. Brady",
    description="A collection of functions to accelerate exploration of a Dataset",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mbrady4/Data-Explorer",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
