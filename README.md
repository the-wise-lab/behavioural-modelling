# Behavioural models

This repository contains a selection of useful functions for modelling of behaviour in learning and decision-making tasks.

All of these functions are written on top of [JAX](jax.readthedocs.io/). As a result, this code may not work on Windows unless using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## Installation

To install this package, clone the repository and run `pip install . -e` from the root directory.

For example:

```bash
git clone https://github.com/the-wise-lab/behavioural-modelling.git
cd behavioural-modelling
pip install . -e
```

It can then be used as a regular Python package.

## Examples

Example Jupyter notebooks can be found in the `examples` directory.

## General principles

The intention of this package to provide functions implementing specific behavioural models, **not** for fitting these models. The idea is that these functions are modular and can be used in a variety of contexts, including fitting, simulation, and analysis.

A scenario case might involve:

1. Defining a model. 
2. Simulating data from that model for multiple trials and multiple subjects.
3. Performing model fitting based on this.

This package is intended to perform step **(1)** only.
