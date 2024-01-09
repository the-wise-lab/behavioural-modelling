import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


@jax.jit
def choice_from_action_p_single(
    key: jax.random.PRNGKey, probs: ArrayLike, lapse: float = 0.0
) -> int:
    """
    Choose an action from a set of action probabilities for a single choice.

    Args:
        key (jax.random.PRNGKey): Jax random key
        probs (ArrayLike): 1D array of action probabilities, of shape (n_actions)
        lapse (float, optional): Lapse parameter. On lapse trials, a random action is selected. Defaults to 0.0.

    Returns:
        int: Chosen action
    """

    # Get number of possible actions
    n_actions = len(probs)

    # Deal with zero values etc
    probs = probs + 1e-6 / jnp.sum(probs)

    # Add noise
    noise = jax.random.uniform(key) < lapse

    # Choose action
    choice = (1 - noise) * jax.random.choice(
        key, jnp.arange(n_actions, dtype=int), p=probs
    ) + noise * jax.random.randint(key, shape=(), minval=0, maxval=n_actions)

    return choice


choice_func_vmap = jax.vmap(choice_from_action_p_single, in_axes=(None, 0, None))


@jax.jit
def choice_from_action_p(key: jax.random.PRNGKey, probs: ArrayLike, lapse: float = 0.0) -> int:
    """
    Choose an action from a set of action probabilities. Can take probabilities
    in the form of an n-dimensional array, where the last dimension is the
    number of actions.

    Noise is added to the choice, with probability `lapse`. This means that
    on "lapse" trials, the subject will choose an action uniformly at random.

    Args:
        key (int): Jax random key
        probs (np.ndarray): N-dimension array of action probabilities, of shape (..., n_actions)
        lapse (float, optional): Probability of lapse. Defaults to 0.0.
    Returns:
        int: Chosen action
    """

    # Reshape probs
    probs_reshaped = probs.reshape((-1, probs.shape[-1]))

    # Get choices
    choices = choice_func_vmap(key, probs_reshaped, lapse)

    # Reshape choices
    choices = choices.reshape(probs.shape[:-1])

    return choices
