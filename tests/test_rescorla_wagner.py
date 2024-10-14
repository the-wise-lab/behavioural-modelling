import jax
import jax.numpy as jnp
from typing import Tuple, Union
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_rescorla_wagner_update,
    asymmetric_rescorla_wagner_update_choice,
)
from behavioural_modelling.utils import choice_from_action_p


# Test Case 1: Positive Prediction Error, Action Chosen (Single Action)
def test_case_1():
    value = jnp.array([0.5])  # Initial value estimate for one action
    outcome = jnp.array([1.0])  # Outcome is higher than value
    chosen = jnp.array([1])  # Action is chosen (one-hot encoded)
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    # Expected calculations
    pe = (outcome - value) * chosen
    alpha_t = alpha_p * (pe > 0) + alpha_n * (pe < 0)
    expected_updated_value = value + alpha_t * pe + choice_stickiness * chosen
    expected_prediction_error = pe

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 1 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 1 Failed"


# Test Case 2: Negative Prediction Error, Action Chosen (Single Action)
def test_case_2():
    value = jnp.array([0.7])
    outcome = jnp.array([0.3])
    chosen = jnp.array([1])
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.0)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    pe = (outcome - value) * chosen
    alpha_t = alpha_p * (pe > 0) + alpha_n * (pe < 0)
    expected_updated_value = value + alpha_t * pe + choice_stickiness * chosen
    expected_prediction_error = pe

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 2 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 2 Failed"


# Test Case 3: Action Not Chosen (Single Action)
def test_case_3():
    value = jnp.array([0.5])
    outcome = jnp.array([0.7])
    chosen = jnp.array([0])  # Action not chosen
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    expected_updated_value = value  # No change expected
    expected_prediction_error = jnp.array([0.0])  # No prediction error

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 3 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 3 Failed"


# Test Case 4: Zero Prediction Error (Single Action)
def test_case_4():
    value = jnp.array([0.5])
    outcome = jnp.array([0.5])
    chosen = jnp.array([1])
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.0)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    expected_updated_value = value  # No change expected
    expected_prediction_error = jnp.array([0.0])

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 4 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 4 Failed"


# Test Case 5: Multiple Actions, One Chosen
def test_case_5():
    value = jnp.array([0.5, 0.3, 0.2])  # Initial values for multiple actions
    outcome = jnp.array([1.0, 0.0, 0.0])  # Outcomes for each action
    chosen = jnp.array([1, 0, 0])  # First action is chosen
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    pe = (outcome - value) * chosen
    alpha_t = alpha_p * (pe > 0) + alpha_n * (pe < 0)
    expected_updated_value = value + alpha_t * pe + choice_stickiness * chosen
    expected_prediction_error = pe

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 5 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 5 Failed"


# Test Case 6: Multiple Actions, No Action Chosen
def test_case_6():
    value = jnp.array([0.5, 0.3, 0.2])
    outcome = jnp.array([1.0, 0.0, 0.0])
    chosen = jnp.array([0, 0, 0])  # No action chosen
    outcome_chosen = (outcome, chosen)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.05)
    choice_stickiness = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, choice_stickiness
        )
    )

    expected_updated_value = value  # No change expected
    expected_prediction_error = jnp.array([0.0, 0.0, 0.0])

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Test Case 6 Failed"
    assert jnp.allclose(
        prediction_error, expected_prediction_error
    ), "Test Case 6 Failed"


# Test Case 1: Basic functionality with fixed random seed
def test_case_1():
    """
    Tests the function with a fixed random seed to ensure reproducibility.
    """
    value = jnp.array([0.5, 0.3, 0.2])  # Initial value estimates
    outcome = jnp.array([1.0, 0.0, 0.0])  # Outcomes for each action
    key = jax.random.PRNGKey(0)  # Fixed random key for reproducibility
    alpha_p = 0.1
    alpha_n = 0.05
    temperature = 1.0
    n_actions = 3
    choice_stickiness = 0.2

    outcome_key = (outcome, key)

    # Call the function
    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value,
            outcome_key,
            alpha_p,
            alpha_n,
            temperature,
            n_actions,
            choice_stickiness,
        )
    )

    # Expected choice probabilities
    logits = value / temperature
    exp_logits = jnp.exp(logits)
    expected_choice_p = exp_logits / exp_logits.sum()

    # Verify choice probabilities
    assert jnp.allclose(
        choice_p, expected_choice_p
    ), "Choice probabilities mismatch"

    # Expected choice (since key is fixed)
    expected_choice = choice_from_action_p(key, expected_choice_p)

    # Expected choice array
    expected_choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    expected_choice_array = expected_choice_array.at[expected_choice].set(1)

    # Verify choice array
    assert jnp.array_equal(
        choice_array, expected_choice_array
    ), "Choice array mismatch"

    # Compute expected prediction error
    expected_prediction_error = (outcome - value) * expected_choice_array

    # Compute alpha_t
    alpha_t = alpha_p * (expected_prediction_error > 0) + alpha_n * (
        expected_prediction_error < 0
    )

    # Compute expected updated value
    expected_updated_value = (
        value
        + alpha_t * expected_prediction_error
        + choice_stickiness * expected_choice_array
    )

    # Verify updated value
    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Updated value mismatch"

    print("Test Case 1 Passed.")


# Test Case 2: Different learning rates
def test_case_2():
    """
    Tests the function with different learning rates for positive and negative prediction errors.
    """
    value = jnp.array([0.6, 0.4, 0.2])
    outcome = jnp.array([0.0, 1.0, 0.0])
    key = jax.random.PRNGKey(42)
    alpha_p = 0.2
    alpha_n = 0.1
    temperature = 0.5
    n_actions = 3
    choice_stickiness = 0.1

    outcome_key = (outcome, key)

    # Call the function
    updated_value, (_, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value,
            outcome_key,
            alpha_p,
            alpha_n,
            temperature,
            n_actions,
            choice_stickiness,
        )
    )

    # Assertions similar to Test Case 1
    logits = value / temperature
    exp_logits = jnp.exp(logits)
    expected_choice_p = exp_logits / exp_logits.sum()

    assert jnp.allclose(
        choice_p, expected_choice_p
    ), "Choice probabilities mismatch"

    expected_choice = choice_from_action_p(key, expected_choice_p)
    expected_choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    expected_choice_array = expected_choice_array.at[expected_choice].set(1)

    assert jnp.array_equal(
        choice_array, expected_choice_array
    ), "Choice array mismatch"

    expected_prediction_error = (outcome - value) * expected_choice_array
    alpha_t = alpha_p * (expected_prediction_error > 0) + alpha_n * (
        expected_prediction_error < 0
    )
    expected_updated_value = (
        value
        + alpha_t * expected_prediction_error
        + choice_stickiness * expected_choice_array
    )

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Updated value mismatch"

    print("Test Case 2 Passed.")


# Test Case 3: High temperature (uniform choice probabilities)
def test_case_3():
    """
    Tests the function with a high temperature to ensure uniform choice probabilities.
    """
    value = jnp.array([0.5, 0.5, 0.5])
    outcome = jnp.array([1.0, 0.0, 0.0])
    key = jax.random.PRNGKey(123)
    alpha_p = 0.1
    alpha_n = 0.05
    temperature = 100.0  # High temperature
    n_actions = 3
    choice_stickiness = 0.0

    outcome_key = (outcome, key)

    updated_value, (_, choice_p, choice_array, _) = (
        asymmetric_rescorla_wagner_update_choice(
            value,
            outcome_key,
            alpha_p,
            alpha_n,
            temperature,
            n_actions,
            choice_stickiness,
        )
    )

    # Expected choice probabilities should be approximately uniform
    expected_choice_p = jnp.array([1 / 3, 1 / 3, 1 / 3])

    assert jnp.allclose(
        choice_p, expected_choice_p, atol=1e-2
    ), "Choice probabilities are not uniform"

    print("Test Case 3 Passed.")


# Test Case 4: Zero temperature (greedy choice)
def test_case_4():
    """
    Tests the function with a zero temperature to ensure greedy choice.
    """
    value = jnp.array([0.7, 0.2, 0.1])
    outcome = jnp.array([0.0, 1.0, 0.0])
    key = jax.random.PRNGKey(456)
    alpha_p = 0.1
    alpha_n = 0.05
    temperature = 0.01  # Near zero temperature
    n_actions = 3
    choice_stickiness = 0.0

    outcome_key = (outcome, key)

    updated_value, (_, choice_p, choice_array, _) = (
        asymmetric_rescorla_wagner_update_choice(
            value,
            outcome_key,
            alpha_p,
            alpha_n,
            temperature,
            n_actions,
            choice_stickiness,
        )
    )

    # Expected choice probabilities should be one for the highest value action
    expected_choice_p = jnp.array([1.0, 0.0, 0.0])

    assert jnp.allclose(
        choice_p, expected_choice_p, atol=1e-2
    ), "Choice probabilities do not indicate greedy choice"

    print("Test Case 4 Passed.")


# Test Case 5: Different choice stickiness values
def test_case_5():
    """
    Tests the function with different choice stickiness values.
    """
    value = jnp.array([0.3, 0.5, 0.2])
    outcome = jnp.array([0.0, 1.0, 0.0])
    key = jax.random.PRNGKey(789)
    alpha_p = 0.1
    alpha_n = 0.05
    temperature = 1.0
    n_actions = 3
    choice_stickiness = 0.5  # High stickiness

    outcome_key = (outcome, key)

    updated_value, (old_value, _, choice_array, _) = (
        asymmetric_rescorla_wagner_update_choice(
            value,
            outcome_key,
            alpha_p,
            alpha_n,
            temperature,
            n_actions,
            choice_stickiness,
        )
    )

    # Compute expected updated value with stickiness
    expected_prediction_error = (outcome - value) * choice_array
    alpha_t = alpha_p * (expected_prediction_error > 0) + alpha_n * (
        expected_prediction_error < 0
    )
    expected_updated_value = (
        value
        + alpha_t * expected_prediction_error
        + choice_stickiness * choice_array
    )

    assert jnp.allclose(
        updated_value, expected_updated_value
    ), "Updated value mismatch with choice stickiness"

    print("Test Case 5 Passed.")
