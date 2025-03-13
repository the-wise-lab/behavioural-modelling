import jax
import numpy as np
import pytest
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_rescorla_wagner_update,
)
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_rescorla_wagner_update_choice,
)
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_volatile_rescorla_wagner_update,
)
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_volatile_dynamic_rescorla_wagner_update_choice,
)
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_volatile_rescorla_wagner_single_value_update_choice,
)
from behavioural_modelling.learning.rescorla_wagner import (
    asymmetric_rescorla_wagner_update_choice_sticky,
)

import jax.numpy as jnp


def test_asymmetric_rescorla_wagner_update_positive_error():
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(1.0), jnp.array(1.0))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.isclose(prediction_error, 0.5)
    assert np.isclose(updated_value, 0.55)


def test_asymmetric_rescorla_wagner_update_negative_error():
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(0.0), jnp.array(1.0))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.isclose(prediction_error, -0.5)
    assert np.isclose(updated_value, 0.4)


def test_asymmetric_rescorla_wagner_update_unchosen_action():
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(1.0), jnp.array(0.0))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.isclose(prediction_error, 0.0)
    assert np.isclose(updated_value, 0.5)


def test_asymmetric_rescorla_wagner_update_zero_error():
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(0.5), jnp.array(1.0))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.isclose(prediction_error, 0.0)
    assert np.isclose(updated_value, 0.5)


def test_asymmetric_rescorla_wagner_update_multiple_values():
    value = jnp.array([0.2, 0.5, 0.8])
    outcome_chosen = (jnp.array([0.0, 1.0, 0.5]), jnp.array([1.0, 1.0, 1.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.allclose(prediction_error, np.array([-0.2, 0.5, -0.3]))
    assert np.allclose(updated_value, np.array([0.16, 0.55, 0.74]))


def test_asymmetric_rescorla_wagner_update_partial_choices():
    value = jnp.array([0.3, 0.7, 0.2])
    outcome_chosen = (jnp.array([1.0, 0.5, 0.0]), jnp.array([1.0, 0.0, 1.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    assert np.allclose(prediction_error, np.array([0.7, 0.0, -0.2]))
    assert np.allclose(updated_value, np.array([0.37, 0.7, 0.16]))


def test_asymmetric_rescorla_wagner_update_varying_alphas():
    # Case 1: alpha_p is large, alpha_n is small
    # Expect greater adjustment for positive error
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(1.0), jnp.array(1.0))
    alpha_p = jnp.array(0.5)
    alpha_n = jnp.array(0.1)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )
    # prediction error = 0.5
    # update = 0.5 + 0.5 * 0.5 = 0.75
    assert np.isclose(updated_value, 0.75)

    # Case 2: alpha_p is small, alpha_n is larger
    # Expect greater adjustment for negative error
    value = jnp.array(0.5)
    outcome_chosen = (jnp.array(0.0), jnp.array(1.0))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.5)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )
    # prediction error = -0.5
    # update = 0.5 + 0.5 * (-0.5) = 0.25
    assert np.isclose(updated_value, 0.25)


def test_asymmetric_rescorla_wagner_update_alpha_sensitivity():
    base_value = jnp.array(0.5)
    outcome_chosen = (jnp.array(1.0), jnp.array(1.0))  # positive error
    alpha_n = jnp.array(0.1)

    # Expect larger alpha_p to yield bigger updates from 0.5 toward 1.0
    alpha_values = [0.1, 0.3, 0.5]
    updated_results = []

    for alpha_p in alpha_values:
        updated_value, _ = asymmetric_rescorla_wagner_update(
            base_value, outcome_chosen, alpha_p, alpha_n
        )
        updated_results.append(float(updated_value))

    # Check monotonic increase with alpha_p
    assert updated_results[0] < updated_results[1] < updated_results[2]


def test_asymmetric_rescorla_wagner_update_choice_positive_error():
    value = jnp.array([0.5, 0.2])
    outcome = jnp.array([1.0, 0.0])
    key = jax.random.PRNGKey(0)
    alpha_p = 0.1
    alpha_n = 0.2
    temperature = 1.0
    n_actions = 2

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, (outcome, key), alpha_p, alpha_n, temperature, n_actions
        )
    )

    assert updated_value.shape == value.shape
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert prediction_error.shape == value.shape


def test_asymmetric_rescorla_wagner_update_choice_negative_error():
    value = jnp.array([0.5, 0.8])
    outcome = jnp.array([0.0, 1.0])
    key = jax.random.PRNGKey(1)
    alpha_p = 0.1
    alpha_n = 0.2
    temperature = 1.0
    n_actions = 2

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, (outcome, key), alpha_p, alpha_n, temperature, n_actions
        )
    )
    print(prediction_error)
    assert updated_value.shape == value.shape
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert prediction_error.shape == value.shape


def test_asymmetric_rescorla_wagner_update_choice_zero_error():
    value = jnp.array([0.5, 0.5])
    outcome = jnp.array([0.5, 0.5])
    key = jax.random.PRNGKey(2)
    alpha_p = 0.1
    alpha_n = 0.2
    temperature = 1.0
    n_actions = 2

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, (outcome, key), alpha_p, alpha_n, temperature, n_actions
        )
    )

    assert updated_value.shape == value.shape
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert prediction_error.shape == value.shape


def test_asymmetric_rescorla_wagner_update_choice_multiple_values():
    value = jnp.array([0.2, 0.5, 0.8])
    outcome = jnp.array([0.0, 1.0, 0.5])
    key = jax.random.PRNGKey(3)
    alpha_p = 0.1
    alpha_n = 0.2
    temperature = 1.0
    n_actions = 3

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, (outcome, key), alpha_p, alpha_n, temperature, n_actions
        )
    )

    assert updated_value.shape == value.shape
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert prediction_error.shape == value.shape


def test_asymmetric_rescorla_wagner_update_choice_varying_alphas():
    # Case 1: alpha_p is large, alpha_n is small
    # Expect greater adjustment for positive error
    value = jnp.array([0.5, 0.5])
    key = jax.random.PRNGKey(0)
    outcome_key = (jnp.array(1.0), key)
    alpha_p = jnp.array(0.5)
    alpha_n = jnp.array(0.1)
    temperature = 1.0

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, outcome_key, alpha_p, alpha_n, temperature, 2
        )
    )
    # prediction error = 0.5
    # update = 0.5 + 0.5 * 0.5 = 0.75
    assert np.isclose(updated_value[choice_array.astype(bool)], 0.75)

    # Case 2: alpha_p is small, alpha_n is larger
    # Expect greater adjustment for negative error
    value = jnp.array([0.5, 0.5])
    key = jax.random.PRNGKey(1)
    outcome_key = (jnp.array(0.0), key)
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.5)
    temperature = 1.0

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_rescorla_wagner_update_choice(
            value, outcome_key, alpha_p, alpha_n, temperature, 2
        )
    )
    # prediction error = -0.5
    # update = 0.5 + 0.5 * (-0.5) = 0.25
    assert np.isclose(updated_value[choice_array.astype(bool)], 0.25)


def test_asymmetric_rescorla_wagner_update_choice_temperature_sensitivity():
    value = jnp.array([0.9, 0.1])
    outcome = jnp.array([1.0, 0.0])
    key = jax.random.PRNGKey(5)
    alpha_p = 0.1
    alpha_n = 0.2
    n_actions = 2

    # Expect different choice probabilities with varying temperatures
    temperatures = [0.1, 1.0, 10.0]
    choice_probabilities = []

    for temperature in temperatures:
        _, (_, choice_p, _, _) = asymmetric_rescorla_wagner_update_choice(
            value, (outcome, key), alpha_p, alpha_n, temperature, n_actions
        )
        choice_probabilities.append(choice_p)

    # Check that choice probabilities vary with temperature
    print(choice_probabilities[0], choice_probabilities[1])
    assert not np.allclose(choice_probabilities[0], choice_probabilities[1])
    assert not np.allclose(choice_probabilities[1], choice_probabilities[2])


def test_asymmetric_volatile_rescorla_wagner_update_stable_positive():
    # Test case 1: Stable condition, positive prediction error
    value = jnp.array([0.5])
    outcome = jnp.array(1.0)
    chosen = jnp.array([1.0])
    volatility = jnp.array(0.0)  # stable
    alpha_base = 0.0  # sigmoid(0.0) = 0.5
    alpha_volatility = 0.0
    alpha_pos_neg = 1.0  # increase learning rate for positive PEs
    alpha_interaction = 0.0

    updated_value, (old_value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    # Expected learning rate ≈ 0.731 (sigmoid(0 + 0 + 1 + 0))
    assert np.isclose(updated_value[0], 0.5 + 0.731 * 0.5, rtol=1e-3)


def test_asymmetric_volatile_rescorla_wagner_update_volatile_negative():
    # Test case 2: Volatile condition, negative prediction error
    value = jnp.array([0.8])
    outcome = jnp.array(0.0)
    chosen = jnp.array([1.0])
    volatility = jnp.array(1.0)  # volatile
    alpha_base = 0.0
    alpha_volatility = 1.0  # increase learning rate for volatile trials
    alpha_pos_neg = 1.0
    alpha_interaction = -0.5

    updated_value, (old_value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    # Prediction error is -0.8
    # Learning rate affected by volatility and negative PE
    assert prediction_error[0] == -0.8
    assert updated_value[0] < value[0]  # Should decrease value


def test_asymmetric_volatile_rescorla_wagner_update_multiple_actions():
    # Test case 3: Multiple actions, only update chosen action
    value = jnp.array([0.5, 0.5])
    outcome = jnp.array(1.0)
    chosen = jnp.array([1.0, 0.0])  # First action chosen
    volatility = jnp.array(0.0)
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0

    updated_value, (old_value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    # Only first value should update, second should remain unchanged
    assert updated_value[0] != value[0]
    assert updated_value[1] == value[1]


def test_asymmetric_volatile_rescorla_wagner_update_interaction():
    # Test case 4: Test interaction effect
    value = jnp.array([0.5])
    outcome = jnp.array(1.0)
    chosen = jnp.array([1.0])
    volatility = jnp.array(1.0)
    alpha_base = 0.0
    alpha_volatility = 1.0
    alpha_pos_neg = 1.0
    alpha_interaction = 2.0  # Strong positive interaction

    updated_value, (old_value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    # Learning rate should be higher due to positive interaction between
    # volatility and positive prediction error
    high_interaction_update = updated_value[0]

    # Compare with lower interaction
    updated_value_low_interaction, _ = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction=0.0,
        )
    )

    # Higher interaction should lead to larger update
    assert abs(high_interaction_update - value[0]) > abs(
        updated_value_low_interaction[0] - value[0]
    )


def test_asymmetric_volatile_rw_base_learning_rate_sensitivity():
    # Test sensitivity to alpha_base across a range of values
    value = jnp.array([0.5])
    outcome = jnp.array(1.0)
    chosen = jnp.array([1.0])
    volatility = jnp.array(0.0)

    # Keep other parameters fixed
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0

    # Test different base learning rates
    base_rates = [-2.0, 0.0, 2.0]  # sigmoid: [0.12, 0.5, 0.88]
    updates = []

    for alpha_base in base_rates:
        updated_value, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
        updates.append(float(updated_value[0]))

    # Check monotonic increase in learning with alpha_base
    assert updates[0] < updates[1] < updates[2]


def test_asymmetric_volatile_rw_volatility_sensitivity():
    # Test sensitivity to alpha_volatility in volatile vs stable conditions
    value = jnp.array([0.5])
    outcome = jnp.array(1.0)
    chosen = jnp.array([1.0])

    # Keep other parameters fixed
    alpha_base = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0

    # Test with different volatility coefficients
    volatility_rates = [0.0, 1.0, 2.0]

    # Compare stable vs volatile conditions
    stable_updates = []
    volatile_updates = []

    for alpha_vol in volatility_rates:
        # Stable condition
        updated_value_stable, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, jnp.array(0.0)),
            alpha_base,
            alpha_vol,
            alpha_pos_neg,
            alpha_interaction,
        )
        stable_updates.append(float(updated_value_stable[0]))

        # Volatile condition
        updated_value_volatile, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, chosen, jnp.array(1.0)),
            alpha_base,
            alpha_vol,
            alpha_pos_neg,
            alpha_interaction,
        )
        volatile_updates.append(float(updated_value_volatile[0]))

    # Stable condition should show no effect of alpha_volatility
    assert np.allclose(stable_updates[0], stable_updates[1], stable_updates[2])

    # Volatile condition should show increasing updates with alpha_volatility
    assert volatile_updates[0] < volatile_updates[1] < volatile_updates[2]


def test_asymmetric_volatile_rw_pos_neg_sensitivity():
    # Test sensitivity to alpha_pos_neg for positive vs negative PEs
    value = jnp.array([0.5])
    chosen = jnp.array([1.0])
    volatility = jnp.array(0.0)

    # Keep other parameters fixed
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_interaction = 0.0

    pos_neg_rates = [-1.0, 0.0, 1.0]

    # Test with positive prediction errors
    positive_updates = []
    negative_updates = []

    for alpha_pn in pos_neg_rates:
        # Positive PE
        updated_value_pos, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (jnp.array(1.0), chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pn,
            alpha_interaction,
        )
        positive_updates.append(float(updated_value_pos[0]))

        # Negative PE
        updated_value_neg, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (jnp.array(0.0), chosen, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pn,
            alpha_interaction,
        )
        negative_updates.append(float(updated_value_neg[0]))

    # Higher alpha_pos_neg should increase positive updates
    assert positive_updates[0] < positive_updates[1] < positive_updates[2]

    # Higher alpha_pos_neg should decrease negative updates (more negative)
    assert negative_updates[0] < negative_updates[1] < negative_updates[2]


def test_asymmetric_volatile_rw_interaction_sensitivity():
    # Test sensitivity to interaction term
    value = jnp.array([0.5])
    chosen = jnp.array([1.0])

    # Keep other parameters fixed
    alpha_base = 0.0
    alpha_volatility = 1.0
    alpha_pos_neg = 1.0

    interaction_rates = [-1.0, 0.0, 1.0]

    # Test different combinations of volatility and PE sign
    updates_volatile_pos = []
    updates_volatile_neg = []

    for alpha_int in interaction_rates:
        # Volatile condition, positive PE
        updated_value_pos, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (jnp.array(1.0), chosen, jnp.array(1.0)),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_int,
        )
        updates_volatile_pos.append(float(updated_value_pos[0]))

        # Volatile condition, negative PE
        updated_value_neg, _ = asymmetric_volatile_rescorla_wagner_update(
            value,
            (jnp.array(0.0), chosen, jnp.array(1.0)),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_int,
        )
        updates_volatile_neg.append(float(updated_value_neg[0]))

    # Higher interaction term should increase updates for volatile positive PEs
    assert (
        updates_volatile_pos[0]
        < updates_volatile_pos[1]
        < updates_volatile_pos[2]
    )
    # Higher interaction term should decrease updates for volatile negative PEs (more negative)
    assert (
        updates_volatile_neg[0]
        < updates_volatile_neg[1]
        < updates_volatile_neg[2]
    )


def test_asymmetric_volatile_dynamic_rw_choice_basic():
    # Test basic choice functionality
    value = jnp.array([0.7, 0.3])
    outcome = jnp.array(1.0)
    key = jax.random.PRNGKey(0)
    volatility = jnp.array(0.0)

    # Fixed parameters
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    n_actions = 2

    # Test with very low temperature (should choose higher value)
    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            0.1,
            2,
        )
    )

    # Check shapes
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert jnp.sum(choice_array) == 1  # One-hot encoding
    assert choice_p[0] > 0.9  # Should strongly prefer first option


def test_asymmetric_volatile_dynamic_rw_temperature_sensitivity():
    # Test how temperature affects choice probabilities
    value = jnp.array([0.7, 0.3])
    outcome = jnp.array(1.0)
    key = jax.random.PRNGKey(0)
    volatility = jnp.array(0.0)

    # Fixed parameters
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    n_actions = 2

    # Test different temperatures
    temperatures = [0.1, 1.0, 10.0]
    choice_probs = []

    for temp in temperatures:
        _, (_, choice_p, _, _) = (
            asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
                value,
                (outcome, key, volatility),
                alpha_base,
                alpha_volatility,
                alpha_pos_neg,
                alpha_interaction,
                temperature=temp,
                n_actions=n_actions,
            )
        )
        choice_probs.append(choice_p)

    # Higher temperature should lead to more uniform probabilities
    prob_diffs = [abs(cp[0] - cp[1]) for cp in choice_probs]
    assert prob_diffs[0] > prob_diffs[1] > prob_diffs[2]


def test_asymmetric_volatile_dynamic_rw_multiple_actions():
    # Test with more than 2 actions
    value = jnp.array([0.3, 0.5, 0.2])
    outcome = jnp.array(1.0)
    key = jax.random.PRNGKey(0)
    volatility = jnp.array(0.0)

    # Fixed parameters
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    n_actions = 3
    temperature = 1.0

    updated_value, (old_value, choice_p, choice_array, prediction_error) = (
        asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            temperature=temperature,
            n_actions=n_actions,
        )
    )

    # Check shapes and properties
    assert choice_p.shape == (n_actions,)
    assert choice_array.shape == (n_actions,)
    assert jnp.sum(choice_array) == 1
    assert jnp.allclose(jnp.sum(choice_p), 1.0)
    assert jnp.all(choice_p >= 0)


def test_asymmetric_volatile_dynamic_rw_deterministic_seed():
    # Test that same seed produces same choices
    value = jnp.array([0.5, 0.5])
    outcome = jnp.array(1.0)
    key = jax.random.PRNGKey(42)
    volatility = jnp.array(0.0)

    # Fixed parameters
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    n_actions = 2
    temperature = 1.0

    # Run twice with same seed
    _, (_, _, choice_array1, _) = (
        asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            temperature=temperature,
            n_actions=n_actions,
        )
    )

    _, (_, _, choice_array2, _) = (
        asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            temperature=temperature,
            n_actions=n_actions,
        )
    )

    # Choices should be identical
    assert jnp.array_equal(choice_array1, choice_array2)


def test_single_value_complementary_options():
    """Test that the two options are properly set up as complementary values"""
    # Initialize test parameters
    value = 0.7
    outcome = 1.0
    key = jax.random.PRNGKey(0)
    volatility = 0.0
    alpha_base = 0.0  # Set to 0 to prevent value updates for this test
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    temperature = 1.0

    # Run function
    _, (_, choice_p, _, _) = (
        asymmetric_volatile_rescorla_wagner_single_value_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            temperature,
        )
    )

    # Check that probabilities sum to 1 (within floating point precision)
    assert jnp.abs(jnp.sum(choice_p) - 1.0) < 1e-6

    # Check that the values used for choice are complementary
    # Higher probability should be assigned to the option with value 0.7
    # compared to the option with value 0.3 (1 - 0.7)
    assert choice_p[0] > choice_p[1]


def test_value_bounds():
    """Test that the updated value stays within [0, 1] bounds"""
    # Initialize test parameters
    key = jax.random.PRNGKey(0)
    volatility = 1.0
    alpha_base = 2.0  # Large learning rate to test bounds
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    temperature = 1.0

    # Test with extreme initial values and outcomes
    test_cases = [
        (0.99, 1.0),  # High value, positive outcome
        (0.01, 0.0),  # Low value, negative outcome
    ]

    for initial_value, outcome in test_cases:
        updated_value, _ = (
            asymmetric_volatile_rescorla_wagner_single_value_update_choice(
                initial_value,
                (outcome, key, volatility),
                alpha_base,
                alpha_volatility,
                alpha_pos_neg,
                alpha_interaction,
                temperature,
            )
        )

        # Check that value stays within [0, 1] bounds
        assert 0.0 <= float(updated_value[0]) <= 1.0


def test_choice_array_format():
    """Test that the choice array is properly formatted as one-hot encoding"""
    # Initialize test parameters
    value = 0.5
    outcome = 1.0
    key = jax.random.PRNGKey(0)
    volatility = 0.0
    alpha_base = 0.0
    alpha_volatility = 0.0
    alpha_pos_neg = 0.0
    alpha_interaction = 0.0
    temperature = 1.0

    # Run function
    _, (_, _, choice_array, _) = (
        asymmetric_volatile_rescorla_wagner_single_value_update_choice(
            value,
            (outcome, key, volatility),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
            temperature,
        )
    )

    # Check that choice array is one-hot encoded
    assert choice_array.shape == (2,)
    assert jnp.sum(choice_array) == 1
    assert jnp.all(jnp.logical_or(choice_array == 0, choice_array == 1))


def test_asymmetric_rescorla_wagner_counterfactual_default():
    value = jnp.array([0.5, 0.5])
    outcome_chosen = (jnp.array(1.0), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n
        )
    )

    # Chosen action should update toward outcome
    assert np.isclose(updated_value[0], 0.55)  # 0.5 + 0.1 * (1.0 - 0.5)
    # Unchosen action should not be updated by default
    assert np.isclose(updated_value[1], 0.5)  # 0.5 + 0.2 * (0.0 - 0.5)

def test_asymmetric_rescorla_wagner_counterfactual_default_multiple_outcomes():
    value = jnp.array([0.5, 0.5])
    outcome_chosen = (jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.5)
    alpha_n = jnp.array(0.5)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value, outcome_chosen, alpha_p, alpha_n, update_all_options=True
        )
    )

    # Chosen action should update toward outcome
    assert np.isclose(updated_value[0], 0.75) 
    assert np.isclose(updated_value[1], 0.25)  

def test_asymmetric_rescorla_wagner_custom_counterfactual_without_updating_all_options():
    value = jnp.array([0.5, 0.5])
    outcome_chosen = (jnp.array(1.0), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: -x  # Opposite of chosen value

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
        )
    )

    # Chosen action updates normally
    assert np.isclose(updated_value[0], 0.55)
    # Unchosen action shouldn't update because update_all_options=False
    assert np.isclose(updated_value[1], 0.5)  # 0.5 + 0.2 * (0.3 - 0.5)


def test_asymmetric_rescorla_wagner_update_all_options():
    value = jnp.array([0.5, 0.5])
    outcome_chosen = (jnp.array(1.0), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: (1 - x) * (1 - y)  # SHould be 0

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    # Both actions should update, even the unchosen one
    assert np.isclose(updated_value[0], 0.55)  # Toward actual outcome
    assert np.isclose(updated_value[1], 0.4)  # Toward counterfactual value


def test_asymmetric_rescorla_wagner_update_all_options_opposite_signed_outcome():
    value = jnp.array([0.0, 0.0])
    outcome_chosen = (jnp.array(1.0), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: -x  # Should be -1

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    # Both actions should update, even the unchosen one
    assert np.isclose(updated_value[0], 0.1)  # Toward actual outcome
    assert np.isclose(updated_value[1], -0.2)  # Toward counterfactual value


def test_asymmetric_rescorla_wagner_scaled_counterfactual():
    value = jnp.array([0.5, 0.5])
    outcome_chosen = (jnp.array(1.0), jnp.array([1.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: 0.25 * x  # 25% of chosen value

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    assert np.isclose(updated_value[0], 0.55)  # Normal update
    assert np.isclose(updated_value[1], 0.45)  # Update toward 0.5


def test_asymmetric_rescorla_wagner_array_update_all_options():
    value = jnp.array([0.5, 0.5, 0.5])
    outcome_chosen = (jnp.array([1.0, 0.7, 0.3]), jnp.array([1.0, 0.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: 0.3 * jnp.ones_like(x)

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    assert np.isclose(updated_value[0], 0.55)  # Toward 1.0
    assert np.isclose(updated_value[1], 0.46)  # Toward 0.3
    assert np.isclose(updated_value[2], 0.46)  # Toward 0.3


def test_asymmetric_rescorla_wagner_array_negative_counterfactual():
    value = jnp.array([0.0, 0.0, 0.0])
    outcome_chosen = (jnp.array([1.0, 0.0, 0.5]), jnp.array([1.0, 0.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: -x  # Opposite of chosen values

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    assert np.isclose(updated_value[0], 0.1)
    assert np.isclose(updated_value[1], 0.0)
    assert np.isclose(updated_value[2], -0.1)


def test_asymmetric_rescorla_wagner_array_scaled_counterfactual():
    value = jnp.array([0.5, 0.5, 0.5])
    outcome_chosen = (jnp.array([1.0, 0.8, 0.6]), jnp.array([1.0, 0.0, 0.0]))
    alpha_p = jnp.array(0.1)
    alpha_n = jnp.array(0.2)
    counterfactual_value = lambda x, y: 0.25 * x  # 25% of chosen values

    updated_value, (old_value, prediction_error) = (
        asymmetric_rescorla_wagner_update(
            value,
            outcome_chosen,
            alpha_p,
            alpha_n,
            counterfactual_value=counterfactual_value,
            update_all_options=True,
        )
    )

    assert np.isclose(updated_value[0], 0.55)
    assert np.isclose(updated_value[1], 0.44)
    assert np.isclose(updated_value[2], 0.43)


def test_asymmetric_rescorla_wagner_update_choice_sticky_basic():
    """Test basic functionality of sticky choice updating"""
    # Initialize test values
    key = jax.random.PRNGKey(0)
    value = jnp.array([0.5, 0.5])  # Equal initial values
    prev_choice = jnp.array([1, 0])  # Previously chose first option
    outcome = 1.0
    alpha_p = 0.5
    alpha_n = 0.5
    temperature = 1.0
    stickiness = 1.0  # Strong stickiness
    n_actions = 2

    # Run update
    (new_value, new_choice), (
        old_value,
        choice_p,
        choice_array,
        prediction_error,
    ) = asymmetric_rescorla_wagner_update_choice_sticky(
        (value, prev_choice),
        (outcome, key),
        alpha_p,
        alpha_n,
        temperature,
        stickiness,
        n_actions,
    )

    # Basic assertions
    assert new_value.shape == (2,)
    assert choice_p.shape == (2,)
    assert choice_array.shape == (2,)
    assert jnp.sum(choice_p) == pytest.approx(1.0)  # Probabilities sum to 1


def test_stickiness_effect():
    """Test that positive stickiness increases probability of repeating choices"""
    key = jax.random.PRNGKey(0)
    value = jnp.array([0.5, 0.5])  # Equal values
    prev_choice = jnp.array([1, 0])  # Previously chose first option
    outcome = 0.5

    # Run with no stickiness
    (_, _), (_, choice_p_no_sticky, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice),
            (outcome, key),
            0.5,
            0.5,
            1.0,
            0.0,
            2,  # No stickiness
        )
    )

    # Run with positive stickiness
    (_, _), (_, choice_p_sticky, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice),
            (outcome, key),
            0.5,
            0.5,
            1.0,
            2.0,
            2,  # Strong positive stickiness
        )
    )

    # Probability of repeating previous choice should be higher with stickiness
    assert choice_p_sticky[0] > choice_p_no_sticky[0]


def test_negative_stickiness():
    """Test that negative stickiness decreases probability of repeating choices"""
    key = jax.random.PRNGKey(0)
    value = jnp.array([0.5, 0.5])
    prev_choice = jnp.array([1, 0])
    outcome = 0.5

    # Run with negative stickiness
    (_, _), (_, choice_p_neg_sticky, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice),
            (outcome, key),
            0.5,
            0.5,
            1.0,
            -2.0,
            2,  # Strong negative stickiness
        )
    )

    # Run with no stickiness
    (_, _), (_, choice_p_no_sticky, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice), (outcome, key), 0.5, 0.5, 1.0, 0.0, 2
        )
    )

    # Probability of repeating previous choice should be lower with negative stickiness
    assert choice_p_neg_sticky[0] < choice_p_no_sticky[0]


def test_learning_rates():
    """Test that positive and negative learning rates work correctly"""
    key = jax.random.PRNGKey(0)
    value = jnp.array([0.5, 0.5])
    prev_choice = jnp.array([1, 0])

    # Test positive learning rate
    (new_value_pos, choice_array), _ = asymmetric_rescorla_wagner_update_choice_sticky(
        (value, prev_choice),
        (1.0, key),  # Positive outcome
        0.5,
        0.1,
        1.0,
        0.0,
        2,  # Higher positive learning rate
    )

    # Test negative learning rate
    (new_value_neg, choice_array), _ = asymmetric_rescorla_wagner_update_choice_sticky(
        (value, prev_choice),
        (0.0, key),  # Negative outcome
        0.1,
        0.5,
        1.0,
        0.0,
        2,  # Higher negative learning rate
    )

    # Verify learning rate effects
    assert new_value_pos[choice_array.astype(bool)] == 0.75
    assert new_value_neg[choice_array.astype(bool)] == 0.25


def test_temperature_effect():
    """Test that temperature affects choice randomness"""
    key = jax.random.PRNGKey(0)
    value = jnp.array([0.6, 0.4])  # Slightly different values
    prev_choice = jnp.array([0, 0])  # No previous choice effect
    outcome = 0.5

    # Run with high temperature (more random)
    (_, _), (_, choice_p_high_temp, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice),
            (outcome, key),
            0.5,
            0.5,
            5.0,
            0.0,
            2,  # High temperature
        )
    )

    # Run with low temperature (more deterministic)
    (_, _), (_, choice_p_low_temp, _, _) = (
        asymmetric_rescorla_wagner_update_choice_sticky(
            (value, prev_choice),
            (outcome, key),
            0.5,
            0.5,
            0.1,
            0.0,
            2,  # Low temperature
        )
    )

    # High temperature should lead to more similar probabilities
    print(choice_p_high_temp)
    print(choice_p_low_temp)
    high_temp_diff = jnp.abs(choice_p_high_temp[0] - choice_p_high_temp[1])
    low_temp_diff = jnp.abs(choice_p_low_temp[0] - choice_p_low_temp[1])
    assert high_temp_diff < low_temp_diff
