
# Behavioural models

[![Tests](https://github.com/the-wise-lab/behavioural-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/the-wise-lab/behavioural-modelling/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://behavioural-modelling.thewiselab.org)
[![Python](https://img.shields.io/badge/python-≥3.8-blue.svg)](https://www.python.org)

This repository contains a selection of useful functions for modelling of behaviour in learning and decision-making tasks.

All of these functions are written on top of [JAX](jax.readthedocs.io/). As a result, this code may not work on Windows unless using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## Installation

You can install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/the-wise-lab/behavioural-modelling.git
```

Alternatively, clone the repository and run `pip install . -e` from the root directory.

For example:

```bash
git clone https://github.com/the-wise-lab/behavioural-modelling.git
cd behavioural-modelling
pip install -e . 
```

This will install an editable version of the package, meaning that you can modify the code and the changes will be reflected in the package.

It can then be used as a regular Python package:

```python
from behavioural_modelling.decision_rules import softmax
```

## Examples

Example Jupyter notebooks can be found in the `examples` directory.

## General principles

The intention of this package to provide functions implementing specific behavioural models, **not** for fitting these models. The idea is that these functions are modular and can be used in a variety of contexts, including fitting, simulation, and analysis.

A scenario case might involve:

1. Defining a model.
2. Simulating data from that model for multiple trials and multiple subjects.
3. Performing model fitting based on this.

This package is intended to perform step **(1)** only.
