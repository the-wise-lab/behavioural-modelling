from behavioural_modelling.utils import choice_from_action_p
import jax
import numpy as np
import jax.numpy as jnp


def test_choice_from_action_p():
    key = jax.random.PRNGKey(0)

    rng = np.random.default_rng(12345)

    # Generate random probabilities
    probs = rng.uniform(size=(40, 100, 3))

    choices = choice_from_action_p(key, probs)

    # Check that choices line up with probabilities
    assert np.argmax(probs[..., :][(choices == 0)].mean(axis=0)) == 0
    assert np.argmax(probs[..., :][(choices == 1)].mean(axis=0)) == 1
    assert np.argmax(probs[..., :][(choices == 2)].mean(axis=0)) == 2
