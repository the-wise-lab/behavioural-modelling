import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple


def get_state_action_values(
    sas: jnp.ndarray, reward: jnp.ndarray, discount: float, values: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the value of each action for all states in a vectorized manner.

    Args:
        sas (jnp.ndarray): State-action-state transition probabilities of shape
            (n_states, n_actions, n_states)
        reward (jnp.ndarray): Reward at each state
        discount (float): Discount factor
        values (jnp.ndarray): Current estimate of value function

    Returns:
        jnp.ndarray: Estimated action values of shape (n_states, n_actions)
    """
    # Compute the expected reward for each state-action pair
    expected_rewards = reward + discount * values  # shape (n_states,)
    action_values = jnp.einsum(
        "san,n->sa", sas, expected_rewards
    )  # shape (n_states, n_actions)
    return action_values


def state_value_iterator(
    values: jnp.ndarray,
    reward: jnp.ndarray,
    discount: float,
    sas: jnp.ndarray,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    Performs one iteration of the value iteration algorithm.

    Args:
        values (jnp.ndarray): Current estimate of the value function
        reward (jnp.ndarray): Reward at each state
        discount (float): Discount factor
        sas (jnp.ndarray): State-action-state transition probabilities

    Returns:
        Tuple[jnp.ndarray, float, jnp.ndarray]: Updated value function, delta,
        and action values
    """
    old_values = values
    action_values = get_state_action_values(sas, reward, discount, values)

    # Identify valid actions
    valid_actions = sas.sum(axis=2) > 0  # shape (n_states, n_actions)

    # Mask invalid actions
    action_values = jnp.where(valid_actions, action_values, -jnp.inf)

    # Standard value iteration using max over actions
    values = jnp.max(action_values, axis=1)

    # Compute delta as the maximum change across all states
    delta = jnp.max(jnp.abs(values - old_values))

    q_values = action_values

    return values, delta, q_values


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
        reward_function (jnp.ndarray): Reward function (i.e., reward at each state)
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
