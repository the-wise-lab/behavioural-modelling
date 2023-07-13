import jax.numpy as jnp
from behavioural_modelling.learning.beta_models import (
    beta_mean_var,
    multiply_beta_by_scalar,
    sum_betas,
    average_betas,
    leaky_beta_update,
)
from scipy.stats import beta
import numpy as np


def test_beta_mean_var_values():
    # Check that mean and variance are correct for a selection of beta distributions
    # by comparing to scipy.stats.beta (which should be correct...)
    rng = np.random.default_rng(12345)
    for _ in range(100):
        a = rng.uniform(1, 10)
        b = rng.uniform(1, 10)
        mean_scipy, var_scipy = beta.stats(a, b, moments="mv")
        mean, var = beta_mean_var(np.array([a, b]))
        assert np.allclose(mean_scipy, mean)
        assert np.allclose(var_scipy, var)


def test_beta_mean_var_shape():
    # Generate a range of beta a and b values
    rng = np.random.default_rng(12345)

    # Loop over these
    for i in range(100):
        # Generate a range of beta a and b values of a random shape
        ab1 = rng.uniform(1, 10, (rng.integers(1, 100), 2))

        # Use beta_mean_var
        mean1, var1 = beta_mean_var(ab1)

        # Check that shape is correct
        assert mean1.shape == (ab1.shape[0],)
        assert var1.shape == (ab1.shape[0],)


def test_multiply_beta_by_scalar_values():
    # This should produce the correct mean and variance

    # Generate a range of beta a and b values
    rng = np.random.default_rng(12345)
    a = rng.uniform(1, 10, 100)
    b = rng.uniform(1, 10, 100)

    # And for the scalar
    scalar = rng.uniform(0, 1, 100)

    # Loop over these
    for i in range(100):
        # Get empirical mean and variance for truncated beta distribution using numpy
        draws = np.random.beta(a[i], b[i], 10000) * scalar[i]

        # Get mean and variance using beta_mean_var
        mean = draws.mean()
        var = draws.var()

        # Use multiply_beta_by_scalar
        scaled_beta = multiply_beta_by_scalar(np.array([a[i], b[i]]), scalar[i])

        # Get mean and variance using beta_mean_var
        mean2, var2 = beta_mean_var(scaled_beta)

        # Check that mean and variance are correct
        assert np.allclose(mean, mean2, atol=1e-2)
        assert np.allclose(var, var2, atol=1e-2)


def test_multiply_beta_by_scalar_shape():
    # Generate a range of beta a and b values
    rng = np.random.default_rng(12345)

    # Scalar values
    scalar = rng.uniform(0, 1, 100)

    # Loop over these
    for i in range(100):
        # Generate a range of beta a and b values of a random shape
        ab = rng.uniform(1, 10, (rng.integers(1, 100), 2))

        # Use multiply_beta_by_scalar
        scaled_beta = multiply_beta_by_scalar(ab, scalar[i])

        # Check that the shape is correct
        assert scaled_beta.shape == ab.shape


def test_sum_betas():
    # Generate a range of beta a and b values
    rng = np.random.default_rng(12345)

    # Loop over these
    for i in range(100):
        # Generate a range of beta a and b values of a random shape
        ab1 = rng.uniform(1, 10, (rng.integers(1, 100), 2))
        ab2 = rng.uniform(1, 10, ab1.shape)

        # Use multiply_beta_by_scalar
        sum_dist = sum_betas(ab1, ab2)

        # Check that the shape is correct
        assert sum_dist.shape == ab1.shape

        # Check mean and variance
        assert np.allclose(
            beta_mean_var(sum_dist)[0],
            beta_mean_var(ab1)[0] + beta_mean_var(ab2)[0],
            atol=1e-6,
        )
        assert np.allclose(
            beta_mean_var(sum_dist)[1],
            beta_mean_var(ab1)[1] + beta_mean_var(ab2)[1],
            atol=1e-6,
        )


def test_average_betas():
    # Generate a range of beta a and b values
    rng = np.random.default_rng(12345)

    # Loop over these
    for i in range(100):
        # Generate a range of beta a and b values of a random shape
        ab1 = rng.uniform(1, 10, (rng.integers(1, 100), 2))
        ab2 = rng.uniform(1, 10, ab1.shape)

        # Get means and vars
        mean1, var1 = beta_mean_var(ab1)
        mean2, var2 = beta_mean_var(ab2)

        # Use multiply_beta_by_scalar
        avg_dist = average_betas(ab1, ab2)

        # Check that the shape is correct
        assert avg_dist.shape == ab1.shape

        # Check mean and variance
        # NOTE that this is only approximate, since the mean and variance of the average of two beta distributions
        # may not be the same as the average of the means and variances of the two beta distributions
        # However, it should be somewhere nearby
        assert np.allclose(
            beta_mean_var(avg_dist)[0],
            np.mean(np.stack([mean1, mean2]), axis=0),
            atol=0.2,
        )
        assert np.allclose(
            beta_mean_var(avg_dist)[1],
            np.mean(np.stack([var1, var2]), axis=0),
            atol=0.1,
        )


def test_average_betas_W():
    a1, b1 = (6, 1)
    a2, b2 = (1, 6)

    for w in np.linspace(0, 1, 10):
        mean1, var1 = beta_mean_var(np.array([[a1, b1]]))
        mean2, var2 = beta_mean_var(np.array([[a2, b2]]))

        mean_3 = w * mean1 + (1 - w) * mean2
        var_3 = w * var1 + (1 - w) * var2

        # Get average
        avg = average_betas(np.array([[a1, b1]]), np.array([[a2, b2]]), w, 1 - w)

        avg_mean, avg_var = beta_mean_var(avg)

        assert np.isclose(mean_3, avg_mean, atol=0.1)
        assert np.isclose(var_3, avg_var, atol=0.1)


def test_leaky_beta_update_decay():
    beta_params = np.ones((3, 2)) * 2

    choice = np.zeros(3)  # no options chosen
    outcome = 1

    lambdas = np.linspace(0, 1, 10)

    for lambda_ in lambdas:
        lambda_ = 0.9  # can't use plain lambda as it's a reserved keyword in python
        tau_p = 0.5
        tau_n = 0.1

        updated = leaky_beta_update(
            beta_params,
            choice,
            outcome,
            tau_p,
            tau_n,
            lambda_,
        )

        assert np.all(updated == 1 + (lambda_ * 1))


def test_leaky_beta_update_tau_p():
    beta_params = np.ones((3, 2)) * 2

    choice = np.zeros(3)
    choice[1] = 1

    outcome = 1

    lambda_ = 0.9  # can't use plain lambda as it's a reserved keyword in python
    tau_p = 0.5
    tau_n = 0.1

    updated = leaky_beta_update(
        beta_params,
        choice,
        outcome,
        tau_p,
        tau_n,
        lambda_,
    )

    assert updated[1, 0] == 2.4
    # check that all other values are == 1.9
    assert np.all(updated[0, :] == 1.9)
    assert np.all(updated[2, :] == 1.9)
    assert updated[1, 1] == 1.9


def test_leaky_beta_update_tau_n():
    beta_params = np.ones((3, 2)) * 2

    choice = np.zeros(3)
    choice[1] = 1

    outcome = 0

    lambda_ = 0.9  # can't use plain lambda as it's a reserved keyword in python
    tau_p = 0.5
    tau_n = 0.1

    updated = leaky_beta_update(
        beta_params,
        choice,
        outcome,
        tau_p,
        tau_n,
        lambda_,
    )

    assert updated[1, 1] == 2
    # check that all other values are == 1.9
    assert np.all(updated[0, :] == 1.9)
    assert np.all(updated[2, :] == 1.9)
    assert updated[1, 0] == 1.9


def test_leaky_beta_increment():
    beta_params = np.ones((3, 2)) * 2

    choice = np.zeros(3)
    choice[1] = 1
    outcome = 1

    lambdas = np.linspace(0, 1, 10)

    for lambda_ in lambdas:
        lambda_ = 0.9  # can't use plain lambda as it's a reserved keyword in python
        tau_p = 0.5
        tau_n = 0.1

        updated = leaky_beta_update(
            beta_params, choice, outcome, tau_p, tau_n, lambda_, increment=0
        )

        # Values should only be decayed, not incremented
        assert np.all(updated == 1 + (lambda_ * 1))


def test_leaky_beta_update():
    beta_params = np.ones((3, 2)) * 2

    choice = np.zeros(3)
    choice[1] = 1
    outcome = 1

    lambdas = np.linspace(0, 1, 10)

    for lambda_ in lambdas:
        lambda_ = 0.9  # can't use plain lambda as it's a reserved keyword in python
        tau_p = 0.5
        tau_n = 0.1

        updated = leaky_beta_update(
            beta_params, choice, outcome, tau_p, tau_n, lambda_, update=0
        )

        # Values should only be decayed, not incremented
        assert np.all(updated == beta_params)
