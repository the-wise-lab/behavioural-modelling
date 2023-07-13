import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


@jax.jit
def softmax(value: ArrayLike, temperature: float = 1) -> ArrayLike:
    """
    Softmax function, with optional temperature parameter.

    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range 0 > inf. Defaults to 1.

    Returns:
        ArrayLike: Choice probabilities, of shape (n_trials, n_bandits)
    """

    return (jnp.exp(value / temperature)) / (
        jnp.sum(jnp.exp(value / temperature), axis=1)[:, None]
    )


@jax.jit
def softmax_subtract_max(value: ArrayLike, temperature: float = 1) -> ArrayLike:
    """
    Softmax function, with optional temperature parameter.

    Subtracts the maximum value before applying softmax to avoid overflow.

    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range 0 > inf. Defaults to 1.

    Returns:
        ArrayLike: Choice probabilities, of shape (n_trials, n_bandits)
    """
    # Subtract max value to avoid overflow
    return (jnp.exp((value - value.max(axis=1)[:, None]) / temperature)) / (
        jnp.sum(jnp.exp((value - value.max(axis=1)[:, None]) / temperature), axis=1)[
            :, None
        ]
    )
