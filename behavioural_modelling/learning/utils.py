import jax
import jax.numpy as jnp

def complement_counterfactual(reward, chosen):
    """
    Counterfactual function that sets the value of unchosen actions to the
    complement of the value of chosen actions.

    Args:
        reward (jax.typing.ArrayLike): The reward received.
        chosen (jax.typing.ArrayLike): A binary array indicating which
            action(s) were chosen.

    Returns:
        jax.typing.ArrayLike: The counterfactual value.
    """
    return jnp.where(
        chosen == 1, 
        0.0, 
        1.0 - jnp.sum(reward * jnp.asarray(chosen == 1, dtype=reward.dtype))
    )
