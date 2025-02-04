import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from typing import Optional


@jax.jit
def softmax(value: ArrayLike, temperature: float = 1) -> ArrayLike:
    """
    Softmax function, with optional temperature parameter.

    In equation form, this is:

    $$
    P(a) = \\frac{e^{Q(a) / \\tau}}{\\sum_{b} e^{Q(b) / \\tau}}
    $$

    Where `P(a)` is the probability of choosing action `a`,
    `Q(a)` is the value of action `a`, and `\tau` is the
    temperature parameter.

    Note that the value of the temperature parameter will
    depend on the range of the values of the Q function.

    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape
            (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range [0, inf].
            Note that this is temperature rather than inverse temperature;
            values are multipled by this value. Defaults to 1.

    Returns:
        ArrayLike: Choice probabilities, of shape (n_trials, n_bandits)
    """

    return (jnp.exp(value / temperature)) / (
        jnp.sum(jnp.exp(value / temperature), axis=1)[:, None]
    )

@jax.jit
def softmax_inverse_temperature(
    value: ArrayLike, inverse_temperature: float = 1
) -> ArrayLike:
    """
    Softmax function, with optional inverse temperature parameter.

    In equation form, this is:

    $$
    P(a) = \\frac{e^{\\beta \\cdot Q(a)}}{\\sum_{b} e^{\\beta \\cdot Q(b)}}
    $$

    Where `P(a)` is the probability of choosing action `a`,
    `Q(a)` is the value of action `a`, and `beta` is the
    inverse temperature parameter.

    Note that the value of the inverse temperature parameter will
    depend on the range of the values of the Q function.

    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape
            (n_trials, n_bandits)
        inverse_temperature (float, optional): Softmax inverse temperature, in
            range [0, inf]. Note that this is inverse temperature rather than
            temperature; values are multiplied by this value. Defaults to 1.
    """
    return (jnp.exp(inverse_temperature * value)) / (
        jnp.sum(jnp.exp(inverse_temperature * value), axis=1)[:, None]
    )

@jax.jit
def softmax_stickiness(
    value: ArrayLike,
    temperature: float = 1.0,
    stickiness: float = 0.0,
    prev_choice: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Softmax function with choice stickiness, and optional temperature
    parameter.

    The standard softmax function is:

    $$
    P(a) = \\frac{e^{Q(a) / \\tau}}{\\sum_{b} e^{Q(b) / \\tau}}
    $$

    With stickiness added:

    $$
    P(a) = \\frac{e^{(Q(a) + \\kappa \\cdot same(a, a_{t-1}))/\\tau}}
    {\\sum_{b} e^{(Q(b) + \\kappa \\cdot same(b, a_{t-1}))/\\tau}}
    $$

    Where:
    - P(a) is the probability of choosing action a
    - Q(a) is the value of action a
    - beta is the temperature parameter
    - kappa is the stickiness parameter
    - same(a, a_{t-1}) is 1 if a matches the previous choice, 0 otherwise

    Args:
        value (ArrayLike): Array of values to apply softmax to, shape
            (n_bandits, )
        temperature (float, optional): Softmax temperature, in range [0, inf].
            Note that this is temperature rather than inverse temperature;
            values are multipled by this value. Defaults to 1.0.
        stickiness (float, optional): Weight given to previous choices, range
            (-inf, inf). Positive values increase probability of repeating
            choices. Defaults to 0.0
        prev_choice (ArrayLike, optional): One-hot encoded previous choices,
            shape (n_bandits, ). Defaults to None.

    Returns:
        ArrayLike: Choice probabilities, shape (n_trials, n_bandits)
    """

    sticky_value = value + stickiness * prev_choice

    return (jnp.exp(temperature * sticky_value)) / (
        jnp.sum(jnp.exp(temperature * sticky_value), axis=1)[:, None]
    )

@jax.jit
def softmax_stickiness_inverse_temperature(
    value: ArrayLike,
    inverse_temperature: float = 1.0,
    stickiness: float = 0.0,
    prev_choice: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Softmax function with choice stickiness, and optional inverse temperature
    parameter.

    The standard softmax function is:

    $$
    P(a) = \\frac{e^{\\beta \\cdot Q(a)}}{\\sum_{b} e^{\\beta \\cdot Q(b)}}
    $$

    With stickiness added:
    
    $$
    P(a) = \\frac{e^{(Q(a) + \\kappa \\cdot same(a, a_{t-1}))/\\tau}}
    {\\sum_{b} e^{(Q(b) + \\kappa \\cdot same(b, a_{t-1}))/\\tau}}
    $$

    Where:
    - P(a) is the probability of choosing action a
    - Q(a) is the value of action a
    - beta is the inverse temperature parameter
    - kappa is the stickiness parameter
    - same(a, a_{t-1}) is 1 if a matches the previous choice, 0 otherwise

    Args:
        value (ArrayLike): Array of values to apply softmax to, shape
            (n_bandits, )
        inverse_temperature (float, optional): Softmax inverse temperature,
            range [0, inf]. Higher values make choices more deterministic.
            Defaults to 1.0
        stickiness (float, optional): Weight given to previous choices, range
            (-inf, inf). Positive values increase probability of repeating
            choices. Defaults to 0.0
        prev_choice (ArrayLike, optional): One-hot encoded previous choices,
            shape (n_bandits, ). Defaults to None.

    Returns:
        ArrayLike: Choice probabilities, shape (n_trials, n_bandits)
    """

    sticky_value = value + stickiness * prev_choice

    return (jnp.exp(inverse_temperature * sticky_value)) / (
        jnp.sum(jnp.exp(inverse_temperature * sticky_value), axis=1)[:, None]
    )


@jax.jit
def softmax_subtract_max(
    value: ArrayLike, temperature: float = 1
) -> ArrayLike:
    """
    Softmax function, with optional temperature parameter.

    Subtracts the maximum value before applying softmax to avoid overflow.

    In equation form, this is:

    $$
    P(a) = \\frac{e^{(Q(a) - \max_{b} Q(b)) / \\tau}}
    {\\sum_{b} e^{(Q(b) - \max_{c} Q(c)) / \\tau}}
    $$

    Where `P(a)` is the probability of choosing action `a`,
    `Q(a)` is the value of action `a`, and `\tau` is the
    temperature parameter.

    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape
            (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range [0, inf].
            Note that this is temperature rather than inverse temperature;
            values are multipled by this value. Defaults to 1.

    Returns:
        ArrayLike: Choice probabilities, of shape (n_trials, n_bandits)
    """
    # Subtract max value to avoid overflow
    return (jnp.exp((value - value.max(axis=1)[:, None]) / temperature)) / (
        jnp.sum(
            jnp.exp((value - value.max(axis=1)[:, None]) / temperature), axis=1
        )[:, None]
    )

