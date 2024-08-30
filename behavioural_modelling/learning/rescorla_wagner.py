import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from ..decision_rules import softmax
from ..utils import choice_from_action_p


@jax.jit
def asymmetric_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome_chosen: Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
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
        outcome_chosen (Tuple[float, float]): A tuple containing the actual
            outcome and a binary value indicating whether the action was
            chosen.
        alpha_p (jax.typing.ArrayLike): The learning rate used when the
            prediction error is positive.
        alpha_n (jax.typing.ArrayLike): The learning rate used when the
            prediction error is negative.

    Returns:
        Tuple[float, float]: The updated value and the prediction error.
    """

    # Unpack the outcome and the chosen action
    outcome, chosen = outcome_chosen

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
    updated_value = value + alpha_t * prediction_error

    return updated_value, (value, prediction_error)


def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome_key: Tuple[jax.typing.ArrayLike, jax.random.PRNGKey],
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
) -> np.ndarray:
    """
    Updates the value estimate using the asymmetric Rescorla-Wagner
    algorithm, and chooses an option based on the softmax function.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome_key (Tuple[jax.typing.ArrayLike, jax.random.PRNGKey]):
            A tuple containing the outcome and the PRNG key.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.

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

    # Unpack outcome and key
    outcome, key = outcome_key

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
        (outcome, choice_array),
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)


asymmetric_rescorla_wagner_update_choice = jax.jit(
    asymmetric_rescorla_wagner_update_choice, static_argnums=(5,)
)
