import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from ..decision_rules import softmax
from ..utils import choice_from_action_p


@jax.jit
def asymmetric_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome_chosen: Tuple[Union[float, jax.typing.ArrayLike], jax.typing.ArrayLike],
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
        outcome_chosen (Tuple[Union[float, jax.typing.ArrayLike],
            jax.typing.ArrayLike]): A tuple containing the actual outcome
            (either as an array or a single value) and a binary array
            indicating which action(s) were chosen. 
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
    updated_value, (_, prediction_error) = asymmetric_rescorla_wagner_update(
        value,
        (outcome, choice_array),
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)


asymmetric_rescorla_wagner_update_choice = jax.jit(
    asymmetric_rescorla_wagner_update_choice, static_argnums=(5,)
)


@jax.jit
def asymmetric_volatile_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome_chosen_volatility: Tuple[
        jax.typing.ArrayLike,
        jax.typing.ArrayLike,
        jax.typing.ArrayLike,
    ],
    alpha_base: float,
    alpha_volatility: float,
    alpha_pos_neg: float,
    alpha_interaction: float,
) -> Tuple[
    jax.typing.ArrayLike, Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]
]:
    """
    Updates the estimated value of a state or action using a variant
    of the Rescorla-Wagner learning rule that incorporates adjusting
    the learning rate based on both volatility and prediction error sign.

    Note that learning rates for this function are transformed using a
    sigmoid function to ensure they are between 0 and 1. The raw
    parameter values supplied to the function must therefore be
    unbounded.

    Args:
        value (jax.typing.ArrayLike): The current estimated value of a
            state or action.
        outcome_chosen_volatility (Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
            jax.typing.ArrayLike]): A tuple containing the outcome, the chosen
            action, and the volatility indicator. The outcome is a float or an
            array (e.g., for a single outcome or multiple outcomes). The chosen
            action is a one-hot encoded array of shape (n_actions,) where 1
            indicates the chosen action and 0 indicates the unchosen actions.
            The volatility indicator is a binary value that indicates whether
            the outcome is volatile (1) or stable (0).
        alpha_base (float): The base learning rate.
        alpha_volatility (float): The learning rate adjustment for volatile
            outcomes.
        alpha_pos_neg (float): The learning rate adjustment for positive and
            negative prediction errors.
        alpha_interaction (float): The learning rate adjustment for the
            interaction between volatility and prediction error sign.

    Returns:
        Tuple[jax.typing.ArrayLike, Tuple[jax.typing.ArrayLike,
            jax.typing.ArrayLike]]:
            - updated_value (jax.typing.ArrayLike): The updated value estimate.
            - output_tuple (Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]):
                - value (jax.typing.ArrayLike): The original value estimate.
                - prediction_error (jax.typing.ArrayLike): The prediction
                  error.
    """

    # Unpack the outcome and the chosen action
    outcome, chosen, volatility_indicator = outcome_chosen_volatility

    # Calculate the prediction error
    prediction_error = outcome - value

    # Set prediction error to 0 for unchosen actions
    prediction_error = prediction_error * chosen

    # Determine whether the error is positive (1) or negative (-1)
    PE_sign = jnp.sign(prediction_error)

    # Compute interaction term (volatility_indicator * error_sign)
    interaction_term = volatility_indicator * PE_sign

    # Compute the dynamic learning rate using base, volatility, and interaction terms
    # Remember we can't use if else statements here because JAX doesn't tolerate them
    # Use adjusted learning rates for positive/negative prediction errors
    alpha_t = jax.nn.sigmoid(
        alpha_base
        + alpha_volatility * volatility_indicator
        + alpha_pos_neg * PE_sign
        + alpha_interaction * interaction_term
    )

    # Update the value
    updated_value = value + alpha_t * prediction_error

    return updated_value, (value, prediction_error)


def asymmetric_volatile_dynamic_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome_key_volatility: Tuple[
        jax.typing.ArrayLike, jax.random.PRNGKey, jax.typing.ArrayLike
    ],
    alpha_base: float,
    alpha_volatility: float,
    alpha_pos_neg: float,
    alpha_interaction: float,
    temperature: float,
    n_actions: int,
) -> jnp.ndarray:
    """
    Updates the value estimate using a variant of the Rescorla-Wagner
    learning rule that adjusts learning rate based on volatility
    and prediction error sign, and chooses an option based on the softmax
    function.

    Note that learning rates for this function are transformed using a
    sigmoid function to ensure they are between 0 and 1. The raw
    parameter values supplied to the function must therefore be
    unbounded.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome_key_volatility (Tuple[jax.typing.ArrayLike, jax.random.PRNGKey,
            jax.typing.ArrayLike]): A tuple containing the outcome, the PRNG key,
            and the volatility indicator.
        alpha_base (float): The base learning rate.
        alpha_volatility (float): The learning rate adjustment for volatile
            outcomes.
        alpha_pos_neg (float): The learning rate adjustment for positive and
            negative prediction errors.
        alpha_interaction (float): The learning rate adjustment for the
            interaction between volatility and prediction error sign.
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
    outcome, key, volatility_indicator = outcome_key_volatility

    # Get choice probabilities
    choice_p = softmax(value[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, (value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, choice_array, volatility_indicator),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)


# Jit compile function
asymmetric_volatile_dynamic_rescorla_wagner_update_choice = jax.jit(
    asymmetric_volatile_dynamic_rescorla_wagner_update_choice,
    static_argnums=(7,),
)


def asymmetric_volatile_rescorla_wagner_single_value_update_choice(
    value: jax.typing.ArrayLike,
    outcome_key_volatility: Tuple[
        jax.typing.ArrayLike, jax.random.PRNGKey, jax.typing.ArrayLike
    ],
    alpha_base: float,
    alpha_volatility: float,
    alpha_pos_neg: float,
    alpha_interaction: float,
    temperature: float,
) -> jnp.ndarray:
    """
    Updates the value estimate using the asymmetric volatile dynamic
    Rescorla-Wagner algorithm, and chooses an option based on the softmax
    function.

    This version of the function is designed for cases where the a single
    value is being learnt, and this value is used to determine which of
    two options to choose. In practice, the value of option 1 is learnt,
    and the value of option 2 is set to 1 - value. This is appropriate
    for cases where the value of one option is the complement of the other.

    Note that learning rates for this function are transformed using a
    sigmoid function to ensure they are between 0 and 1. The raw
    parameter values supplied to the function must therefore be
    unbounded.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome_key_volatility (Tuple[jax.typing.ArrayLike, jax.random.PRNGKey,
            jax.typing.ArrayLike]): A tuple containing the outcome, the PRNG key,
            and the volatility indicator.
        alpha_base (float): The base learning rate.
        alpha_volatility (float): The learning rate adjustment for volatile
            outcomes.
        alpha_pos_neg (float): The learning rate adjustment for positive and
            negative prediction errors.
        alpha_interaction (float): The learning rate adjustment for the
            interaction between volatility and prediction error sign.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.

    """

    # Unpack outcome and key
    outcome, key, volatility_indicator = outcome_key_volatility

    # Get values for two actions - the first is the single
    # value being learned. The second is 1 - value, which
    # is used to calculate choice probabilities
    new_two_options = jnp.array([value, 1 - value])

    # Get choice probabilities
    choice_p = softmax(new_two_options[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(2, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, (value, prediction_error) = (
        asymmetric_volatile_rescorla_wagner_update(
            value,
            (outcome, jnp.ones(1, dtype=jnp.int16), volatility_indicator),
            alpha_base,
            alpha_volatility,
            alpha_pos_neg,
            alpha_interaction,
        )
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)


# Jit compile function
asymmetric_volatile_rescorla_wagner_single_value_update_choice = jax.jit(
    asymmetric_volatile_rescorla_wagner_single_value_update_choice,
)
