import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple


def get_state_action_values(
    s: int,
    n_actions: int,
    sas: jnp.ndarray,
    reward: jnp.ndarray,
    discount: float,
    values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the value of each action for a given state. Used within the main
    value iteration loop.

    Reward is typically conceived of as resulting from taking action A in state
    S. Here, we for the sake of simplicity, we assume that the reward results
    from visiting state S' - that is, taking action A in state S isn't
    rewarding in itself, but the reward received is dependent on the reward
    present in state S'.

    Args:
        s (int): State ID
        n_actions (int): Number of possible actions
        sas (np.ndarray): State, action, state transition function
        reward (np.ndarray): Reward available at each state
        discount (float): Discount factor
        values (np.ndarray): Current estimate of value function


    Returns:
        np.ndarray: Estimated value of each state
    """

    def action_update(s, a, sas, reward, discount, values):
        p_sprime = sas[s, a, :]
        return jnp.dot(p_sprime, reward + discount * values)

    action_values = jax.vmap(
        action_update, in_axes=(None, 0, None, None, None, None)
    )(s, jnp.arange(n_actions, dtype=int), sas, reward, discount, values)

    return action_values


def state_value_iterator(
    values: jnp.ndarray,
    reward: jnp.ndarray,
    discount: float,
    sas: jnp.ndarray,
    soft: bool = False,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    Core value iteration function - calculates value function for the MDP and
    returns q-values for each action in each state.

    This function just runs one iteration of the value iteration algorithm.

    "Soft" value iteration can optionally be performed. This essentially
    involves taking the softmax of action values rather than the max, and is
    useful for inverse reinforcement learning (see Bloem & Bambos, 2014).

    Args:
        values (np.ndarray): Current estimate of the value function
        reward (np.ndarray): Reward at each state (i.e. features x reward
            function)
        discount (float): Discount factor
        sas (np.ndarray): State, action, state transition function
        soft (bool, optional): If True, this implements "soft" value iteration
            rather than standard value iteration. Defaults to False.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: Returns new estimate of the value
            function, new delta, and new q_values
    """
    n_states, n_actions = sas.shape[:2]
    q_values = jnp.zeros((n_states, n_actions))

    def scan_fn(values_delta, s):

        values, delta = values_delta

        v = values[s]  # Current value estimate for state `s`
        action_values = get_state_action_values(
            s, n_actions, sas, reward, discount, values
        )

        if not soft:
            new_value = jnp.max(action_values)
        else:
            new_value = jnp.log(jnp.sum(jnp.exp(action_values)) + 1e-200)

        # Update Q-values for state `s`
        q_values_s = action_values

        # Update delta
        delta = jnp.abs(new_value - v)

        # Update value for state `s`
        values = values.at[s].set(new_value)

        return (values, delta), q_values_s

    # Perform the sequential scan
    (new_values, final_delta), all_q_values = jax.lax.scan(
        scan_fn, (values, 0), jnp.arange(n_states)
    )

    # Combine all Q-values into a single array
    q_values = q_values.at[:, :].set(all_q_values)

    return new_values, final_delta, q_values


def solve_value_iteration(
    n_states: int,
    n_actions: int,
    reward_function: jnp.ndarray,
    max_iter: int,
    discount: float,
    sas: jnp.ndarray,
    tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves an MDP using value iteration given a reward function.

    Args:
        n_states (int): Number of states
        n_actions (int): Number of actions
        reward_function (jnp.ndarray): Reward function (i.e., reward at each
            state)
        max_iter (int): Maximum number of iterations
        discount (float): Discount factor
        sas (jnp.ndarray): State-action-state transition probabilities
        tol (float): Tolerance for convergence

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Final value function and action values
            (Q-values)
    """

    values = jnp.zeros(n_states)

    def cond_fun(carry):
        values, delta, iter_num, q_values = carry
        return (delta > tol) & (iter_num < max_iter)

    def body_fun(carry):
        values, _, iter_num, _ = carry
        values, delta, q_values = state_value_iterator(
            values, reward_function, discount, sas
        )
        iter_num += 1
        return values, delta, iter_num, q_values

    delta = jnp.inf
    iter_num = 0
    q_values = jnp.zeros((n_states, n_actions))
    carry = (values, delta, iter_num, q_values)

    values, delta, iter_num, q_values = lax.while_loop(
        cond_fun, body_fun, carry
    )

    # Mask invalid transitions in Q-values
    invalid_transitions = sas.sum(axis=-1) == 0  # shape (n_states, n_actions)
    q_values = jnp.where(invalid_transitions, -jnp.inf, q_values)

    return values, q_values


# jit compile
solve_value_iteration = jax.jit(solve_value_iteration, static_argnums=(0, 1))
