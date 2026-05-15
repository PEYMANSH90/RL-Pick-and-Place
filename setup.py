"""Package setup for pick_place."""

from pathlib import Path
from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="pick-place",
    version="0.1.0",
    description=(
        "RL-based minimum-torque pick-and-place control for the ABB IRB1600 "
        "using TD3 implemented from scratch."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Peyman Shamsi",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "conf*", "scripts*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "sympy>=1.12",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "torch>=2.0",
        "gymnasium>=0.29",
        "roboticstoolbox-python>=1.1",
    ],
    entry_points={
        "console_scripts": [
            "pick-place-train    = scripts.train:main",
            "pick-place-evaluate = scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
