import jax
import jax.numpy as jnp
from typing import Tuple, Union
from .utils import complement_counterfactual


def pearce_hall_update(
    value: jax.typing.ArrayLike,
    alpha: jax.typing.ArrayLike,
    outcome_chosen: Tuple[
        Union[float, jax.typing.ArrayLike], jax.typing.ArrayLike
    ],
    kappa: jax.typing.ArrayLike,
    gamma: jax.typing.ArrayLike,
    counterfactual_value: callable = complement_counterfactual,
    update_all_options: bool = False,
) -> Tuple[
    Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
]:
    """
    Updates the estimated value and learning rate using the Pearce-Hall
    learning rule.

    The Pearce-Hall model updates value based on prediction error scaled by
    a learning rate (alpha), which itself evolves over time based on
    the absolute prediction error.

    $$ V_{t+1} = V_t + \kappa \cdot \alpha_t \cdot (R_t - V_t) $$
    $$ \alpha_{t+1} = \gamma \cdot |R_t - V_t| + (1 - \gamma) \cdot \alpha_t $$

    By default, value estimates are only updated for chosen actions. For unchosen
    actions, the prediction error is set to 0. This can be modified by setting
    `update_all_options=True`, which will update all actions regardless of
    whether they were chosen.

    We also assume that learning rate (alpha) is stimulus-specific, meaning that
    each action has its own learning rate that is updated only when that action
    is chosen. This can be modified by changing the `counterfactual_value` function.
    This allows the model to adaptively adjust learning rates based
    on the history of prediction errors for each action.

    Counterfactual updating can be used to set the value of unchosen actions
    according to a function of the value of chosen actions. This can be useful
    in cases where the value of unchosen actions should be set to a specific
    value, such as the negative of the value of chosen actions. By default this
    function sets the value of unchosen actions to the complement of the value
    of chosen actions:

    ```python
    counterfactual_value: callable = lambda reward, chosen: jnp.where(
        chosen == 1,
        0.0,
        1.0 - jnp.sum(reward * jnp.asarray(chosen == 1, dtype=reward.dtype))
    )
    ```

    Args:
        value (jax.typing.ArrayLike): Current estimated value.
        alpha (jax.typing.ArrayLike): Current learning rate (associability)
            estimate.
        outcome_chosen (Tuple): A tuple containing the outcome (R_t) and
            a binary array indicating which action(s) were chosen.
        kappa (jax.typing.ArrayLike): The learning rate scaling factor.
        gamma (jax.typing.ArrayLike): The learning rate update rate.
        counterfactual_value (callable, optional): Logic for unchosen options.
        update_all_options (bool, optional): Whether to update unchosen options.

    Returns:
        Tuple:
            - (updated_value, updated_alpha): The new state.
            - (value, alpha, prediction_error): Intermediate values for
              logging.
    """
    outcome, chosen = outcome_chosen

    # Calculate counterfactual for unchosen options
    counterfactual = counterfactual_value(outcome, chosen) * (1 - chosen)
    outcome = outcome * chosen + counterfactual

    # Chosen mask adjustment
    chosen = (chosen * (1 - update_all_options)) + (1 * update_all_options)

    # Prediction error
    prediction_error = outcome - value
    # Zero out PE for unchosen if not updating all
    prediction_error = prediction_error * chosen

    # Update Value
    # V_{t+1} = V_t + kappa * alpha_t * PE
    updated_value = value + kappa * alpha * prediction_error

    # Update Associability
    # alpha_{t+1} = gamma * |PE| + (1 - gamma) * alpha_t
    # We update associability only for chosen options (or all if requested)
    # This allows for stimulus-specific learning rates if desired
    abs_pe = jnp.abs(prediction_error)
    updated_alpha = alpha + chosen * (gamma * (abs_pe - alpha))

    return (updated_value, updated_alpha), (value, alpha, prediction_error)


pearce_hall_update = jax.jit(
    pearce_hall_update, static_argnames="counterfactual_value"
)
