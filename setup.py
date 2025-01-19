from setuptools import find_packages, setup

setup(
    name="behavioural_modelling",
    version="0.1.0",
    description="A package for behavioural modelling",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib", 
        "numpy",
    ],
    tests_require=[
        "pytest",
    ],
    extras_require={
        "test": ["pytest"],
    }
)