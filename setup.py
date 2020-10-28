"""A setuptools-based setup module for kairos-yaml."""

from pathlib import Path

from setuptools import find_packages, setup

import sdf

long_description = Path("README.md").read_text()

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Utilities",
    "Typing :: Typed",
]

requirements = [
    req.replace("==", ">=") for req in Path("requirements.txt").read_text().splitlines()
]

setup(
    name="kairos-yaml",
    version=sdf.__version__,
    description=sdf.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isi-vista/kairos-yaml",
    author="CMU and ISI",
    author_email="ahedges@isi.edu",
    license="MIT",
    classifiers=classifiers,
    install_requires=requirements,
    python_requires=">=3.7",
    packages=find_packages(),
    package_data={"sdf": ["ontology.json"]},
)
