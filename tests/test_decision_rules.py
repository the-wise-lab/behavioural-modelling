import pytest
import jax.numpy as jnp
import jax
from typing import List
import sys
from behavioural_modelling.decision_rules import softmax, softmax_subtract_max, softmax_inverse_temperature


def test_softmax():
    # Generate a range of values from -1 to 1 for a single trial
    test_values = jnp.linspace(-1, 1, 100)[:, None]

    # Add 1-value as a second option
    test_values = jnp.hstack((test_values, (2 - (1 + test_values)) - 1))

    # Define temperatures
    temperatures = [0.25, 0.5, 1, 2, 5, 10]

    # Calculate softmax probabilities for each temperature
    softmax_probs = jnp.array(
        [softmax(test_values, temperature=t) for t in temperatures]
    )

    # Check that probabilities sum to 1
    assert jnp.allclose(softmax_probs.sum(axis=2), 1)

    # Check that probabilities are monotonically increasing with value
    assert jnp.all(jnp.diff(softmax_probs[..., 0], axis=-1) > 0)

    # Check that probability curves get flatter with increasing temperature
    assert jnp.all(jnp.diff(jnp.diff(softmax_probs[:, :, 0], axis=-1).sum(axis=-1)) < 0)


def test_softmax_subtract_max():
    # Generate a range of values from -1 to 1 for a single trial
    test_values = jnp.linspace(-1, 1, 100)[:, None]

    # Add 1-value as a second option
    test_values = jnp.hstack((test_values, (2 - (1 + test_values)) - 1))

    # Define temperatures
    temperatures = [0.25, 0.5, 1, 2, 5, 10]

    # Calculate softmax probabilities for each temperature
    softmax_probs = jnp.array(
        [softmax(test_values, temperature=t) for t in temperatures]
    )

    # Check that probabilities sum to 1
    assert jnp.allclose(softmax_probs.sum(axis=2), 1)

    # Check that probabilities are monotonically increasing with value
    assert jnp.all(jnp.diff(softmax_probs[..., 0], axis=-1) > 0)

    # Check that probability curves get flatter with increasing temperature
    assert jnp.all(jnp.diff(jnp.diff(softmax_probs[:, :, 0], axis=-1).sum(axis=-1)) < 0)


def test_softmax_subtract_max_extreme_values():
    # Generate a range of values from -1 to 1 for a single trial
    test_values = jnp.linspace(-999999999, 99999999, 100)[:, None]

    # Add 1-value as a second option
    test_values = jnp.hstack((test_values, (2 - (1 + test_values)) - 1))

    # Define temperatures
    temperatures = [0.25, 0.5, 1, 2, 5, 10]

    # Calculate softmax probabilities for each temperature
    softmax_probs = jnp.array(
        [softmax_subtract_max(test_values, temperature=t) for t in temperatures]
    )

    # Results shouldn't necessary follow a normal-looking curve (e.g., extreme values
    # won't necessarily be increasing as they'll be at the limit) but we shouldn't
    # get any NaNs or Infs

    # Check that probabilities sum to 1
    assert jnp.allclose(softmax_probs.sum(axis=2), 1)

    # Check that there are no NaNs or Infs
    assert jnp.all(jnp.isfinite(softmax_probs))

def test_softmax_inverse_temperature():
    # Generate a range of values from -1 to 1 for a single trial
    test_values = jnp.linspace(-1, 1, 100)[:, None]

    # Add 1-value as a second option
    test_values = jnp.hstack((test_values, (2 - (1 + test_values)) - 1))

    # Define inverse temperatures
    inverse_temperatures = [0.25, 0.5, 1, 2, 5, 10]

    # Calculate softmax probabilities for each inverse temperature
    softmax_probs = jnp.array(
        [softmax_inverse_temperature(test_values, inverse_temperature=t) for t in inverse_temperatures]
    )

    # Check that probabilities sum to 1
    assert jnp.allclose(softmax_probs.sum(axis=2), 1)

    # Check that probabilities are monotonically increasing with value
    assert jnp.all(jnp.diff(softmax_probs[..., 0], axis=-1) >= 0)

    # Check that probability curves get steeper with increasing inverse temperature
    # (opposite of regular softmax temperature)
    assert jnp.all(jnp.diff(jnp.diff(softmax_probs[:, :, 0], axis=-1).sum(axis=-1)) > 0)

