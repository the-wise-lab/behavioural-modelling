import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from typing import Tuple
from jax.experimental import checkify


@jax.jit
def beta_mean_var(beta_params: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculate mean and variance of a beta distribution.

    Args:
        beta_params (ArrayLike): Parameters of the beta distribution. Of shape (n_options, 2),
        where the first dimension represents the number of options (each of which has its own
        beta distribution), and the second dimension represents the alpha and beta parameters
        of each beta distribution.

    Returns:
        tuple[ArrayLike, ArrayLike]: Mean and variance of the beta distribution.
    """
    a, b = beta_params[..., 0], beta_params[..., 1]
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    return mean, var


@jax.jit
def generalised_beta_mean_var(
    alpha: float, beta: float, a: float, b: float
) -> Tuple[float, float]:
    """
    Calculate mean and variance of a generalised beta distribution.

    Args:
        alpha (float): Alpha parameter of the beta distribution.
        beta (float): Beta parameter of the beta distribution.
        a (float): Lower bound of the beta distribution.
        b (float): Upper bound of the beta distribution.

    Returns:
        tuple[float, float]: Mean and variance of the beta distribution.
    """
    mean = ((b - a) * alpha) / (alpha + beta) + a
    var = (alpha * beta * (b - a) ** 2) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return mean, var


@jax.jit
def multiply_beta_by_scalar(beta_params: ArrayLike, scalar: float) -> jnp.ndarray:
    """
    Multiply a beta distribution by a scalar.

    Args:
        beta_params (ArrayLike): Parameters of beta distribution. Of shape (n_options, 2),
        where the first dimension represents the number of options (each of which has its own
        beta distribution), and the second dimension represents the alpha and beta parameters
        of each beta distribution.
        scalar (float): Scalar to multiply beta distribution by.

    Returns:
        jnp.ndarray: New beta distribution parameters, specified as [a, b].
    """

    # Extract parameters
    a = beta_params[..., 0]
    b = beta_params[..., 1]

    # Calculate mean and variance
    mean, var = beta_mean_var(beta_params)

    # Scale mean and variance
    mean = mean * scalar
    var = var * scalar**2

    # Calculate new parameters
    a_new = mean * ((mean * (1 - mean)) / var - 1)
    b_new = (1 - mean) * ((mean * (1 - mean)) / var - 1)

    # Return new parameters
    return jnp.stack([a_new, b_new], axis=-1)


@jax.jit
def sum_betas(beta1_params: ArrayLike, beta2_params: ArrayLike) -> jnp.ndarray:
    """
    Sum two beta distributions. This uses an approximation described in the following paper:

    Pham, T.G., Turkkan, N., 1994. Reliability of a standby system with beta-distributed component lives.
    IEEE Transactions on Reliability 43, 71–75. https://doi.org/10.1109/24.285114

    Where the first two moments of the summed distribution are calculated as follows:

    $$\mu = \mu_1 + \mu_2$$
    $$\sigma^2 = \sigma_1^2 + \sigma_2^2$$

    We then calculate the parameters of the new beta distribution using the following equations:

    $$\alpha = \mu \left( \frac{\mu (1 - \mu)}{\sigma^2} - 1 \right)$$
    $$\beta = (1 - \mu) \left( \frac{\mu (1 - \mu)}{\sigma^2} - 1 \right)$$

    This function assumes that the means of the two beta distributions sum to <=1. If this is not the case,
    the output will be invalid.

    Args:
        beta1_params (ArrayLike): Parameters of the first beta distribution. Of shape (n_options, 2),
        where the first dimension represents the number of options (each of which has its own
        beta distribution), and the sec
        beta2_params (ArrayLike): Parameters of second beta distribution.

    Returns:
        jnp.ndarray: New beta distribution parameters.
    """

    # Extract parameters
    a1 = beta1_params[..., 0]
    b1 = beta1_params[..., 1]
    a2 = beta2_params[..., 0]
    b2 = beta2_params[..., 1]

    # Calculate means and variances
    mean1 = a1 / (a1 + b1)
    var1 = (a1 * b1) / ((a1 + b1) ** 2 * (a1 + b1 + 1))
    mean2 = a2 / (a2 + b2)
    var2 = (a2 * b2) / ((a2 + b2) ** 2 * (a2 + b2 + 1))

    # Sum means and variances
    mean_new = mean1 + mean2
    var_new = var1 + var2

    # Calculate new parameters
    a_new = mean_new * ((mean_new * (1 - mean_new)) / var_new - 1)
    b_new = (1 - mean_new) * ((mean_new * (1 - mean_new)) / var_new - 1)

    # Return new parameters
    return jnp.stack([a_new, b_new], axis=-1)


@jax.jit
def average_betas(
    beta1_params: ArrayLike,
    beta2_params: ArrayLike,
    W1: float = 0.5,
    W2: float = 0.5,
) -> jnp.ndarray:
    """
    Average two beta distributions, weighted by W.

    Args:
        beta1_params (ArrayLike): Parameters of first beta distribution.
        beta2_params (ArrayLike): Parameters of second beta distribution.

    Returns:
        jnp.ndarray: New beta distribution parameters.
    """

    # Extract parameters
    a1 = beta1_params[..., 0]
    b1 = beta1_params[..., 1]
    a2 = beta2_params[..., 0]
    b2 = beta2_params[..., 1]

    # Calculate average
    a_new = (W1 * a1) + (W2 * a2)
    b_new = (W1 * b1) + (W2 * b2)

    # Return new parameters
    return jnp.stack([a_new, b_new], axis=-1)


@jax.jit
def leaky_beta_update(
    estimate: ArrayLike,
    choices: ArrayLike,
    outcome: float,
    tau_p: float,
    tau_n: float,
    decay: float,
    update: int = 1,
    increment: int = 1,
) -> jnp.ndarray:
    """
    Update estimates using the (asymmetric) leaky beta model. 
    
    This models represents the probability of the outcome associated with each option (e.g., bandits in a bandit task)
    as a beta distribution.

    Values are updated according to the following equations:

    \begin{gathered}
    A_i^{t+1}=\lambda \cdot A_i^{t}+outcome_t \cdot \tau^{+}  \\
    B_i^{t+1}=\lambda \cdot B_i^{t}+(1-outcome_t) \cdot \tau^{-}
    \end{gathered}

    This function also allows for updating to be turned off (i.e., the estimate is not updated at all) and for incrementing
    to be turned off (i.e., decay is applied, but the outcome is not registered).

    Only chosen options incremented, but all options decay.

    Args:
        estimate (ArrayLike): Alpha and beta estimates for this trial. Should be an array of shape (n, 2) where
        the first dimension represents the alpha and beta parameters of the beta distribution and the second
        dimension represents the number of option.
        choices (ArrayLike): Choices made in this trial. Should have as many entries as there are options, with
        zeros for non-chosen options and ones for chosen options (i.e., one-hot encoded).
        outcomes (float): Observed outcome for this trial.
        tau_p (float): Update rate for outcomes equal to 1.
        tau_n (float): Update rate for outcomes equal to 0.
        decay (float): Decay rate.
        update (int, optional): Whether to update the estimate. If 0, the estimate is not updated (i.e., no decay is
        applied, and the outcome of the trial does not affect the outcome). Defaults to 1.
        increment (int, optional): Whether to increment the estimate. If 0, the estimate is not incremented but
        decay is applied. Defaults to 1.
    Returns:
        jnp.ndarray: Updated value estimates for this trial, with one entry per bandit.
    """

    # For each parameter, we apply the decay to (previous value - 1) so that we are in effect
    # treating 1 as the baseline value. This is helpful becuase values of < 1 can produce
    # strange-looking distributions (e.g., with joint peaks at 0 and 1). Keeping values
    # > 1 ensures that the baseline distribution (ignoring any evidence we've observed)
    # is a flat distribution between 0 and 1. This also generally aids parameter recovery.

    # Make sure any outcomes > 1 are set to 1
    outcome = jnp.array(outcome > 0, int)

    # Update alpha
    update_1 = (
        1 + (decay * (estimate[:, 0] - 1)) + (tau_p * (choices * outcome) * increment)
    )
    estimate = estimate.at[:, 0].set(
        (update * update_1) + ((1 - update) * estimate[:, 0])
    )

    # Update beta
    update_2 = (
        1
        + (decay * (estimate[:, 1] - 1))
        + (tau_n * (choices * (1 - outcome)) * increment)
    )
    estimate = estimate.at[:, 1].set(
        (update * update_2) + ((1 - update) * estimate[:, 1])
    )

    return estimate
