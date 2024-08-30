import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def asymmetric_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    chosen: jax.typing.ArrayLike,
    alpha_p: jax.typing.ArrayLike,
    alpha_n: jax.typing.ArrayLike,
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    """
    Updates the estimated value of a state or action using the Asymmetric
    Rescorla-Wagner learning rule.

    The function calculates the prediction error as the difference between
    the actual outcome and the current estimated value. It then updates the
    estimated value based on the prediction error and the learning rate,
    which is determined by whether the prediction error is positive or
    negative.

    Value estimates are only updated for chosen actions. For unchosen
    actions, the prediction error is set to 0.

    Args:
        value (jax.typing.ArrayLike): The current estimated value of a
            state or action.
        outcome (jax.typing.ArrayLike): The actual reward received.
        chosen (jax.typing.ArrayLike): Binary indicator of whether the
            action was chosen (1) or not (0).
        alpha_p (jax.typing.ArrayLike): The learning rate used when the
            prediction error is positive.
        alpha_n (jax.typing.ArrayLike): The learning rate used when the
            prediction error is negative.

    Returns:
        Tuple[float, float]: The updated value and the prediction error.
    """

    # Calculate the prediction error
    prediction_error = outcome - value

    # Set prediction error to 0 for unchosen actions
    prediction_error = prediction_error * chosen

    # Set the learning rate based on the sign of the prediction error Remember
    # - we can't use if else statements here because JAX doesn't tolerate them
    alpha_t = (alpha_p * (prediction_error > 0)) + (
        alpha_n * (prediction_error < 0)
    )

    # Update the value
    value = value + alpha_t * prediction_error

    return value, prediction_error


@jax.jit
def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    key: jax.random.PRNGKey,
) -> np.ndarray:
    """
    Updates the value estimate using the asymmetric Rescorla-Wagner
    algorithm, and chooses an option based on the softmax function.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome (jax.typing.ArrayLike): The outcome of the action.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.
        key (jax.random.PRNGKey): The random key for the choice function.

    Returns:
        Tuple[np.ndarray, Tuple[jax.typing.ArrayLike, np.ndarray, int, 
        np.ndarray]]:
            - updated_value (jnp.ndarray): The updated value estimate.
            - output_tuple (Tuple[jax.typing.ArrayLike, np.ndarray, int, 
            np.ndarray]):
                - value (jax.typing.ArrayLike): The original value estimate.
                - choice_p (jnp.ndarray): The choice probabilities.
                - choice (int): The chosen action.
                - choice_array (jnp.ndarray): The chosen action in one-hot
                  format.
    """

    # Get choice probabilities
    choice_p = softmax(value[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, prediction_error = asymmetric_rescorla_wagner_update(
        value,
        choice_array,
        outcome,
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)