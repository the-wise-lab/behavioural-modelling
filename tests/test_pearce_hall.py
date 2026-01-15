import jax
import numpy as np
import pytest
import jax.numpy as jnp
from behavioural_modelling.learning.pearce_hall import pearce_hall_update

# --- Helper for formulaic expectations ---
def expected_ph_update(val, alpha, outcome, chosen, kappa, gamma, prev_alpha=None):
    # This logic mirrors the implementation for verification
    pe = (outcome - val) * chosen
    new_val = val + kappa * alpha * pe
    
    # Logic note: alpha updates based on current PE
    abs_pe = abs(pe)
    # If unchosen, alpha usually doesn't update unless specific logic applies,
    # but in the function logic: updated_alpha = alpha + chosen * (...)
    new_alpha = alpha + chosen * (gamma * (abs_pe - alpha))
    
    return new_val, new_alpha, pe

@pytest.mark.parametrize("kappa, gamma", [
    (0.5, 0.5),
    (0.1, 0.9),
    (1.0, 0.0), # Gamma 0 -> alpha shouldn't change
    (0.0, 1.0), # Kappa 0 -> value shouldn't change
])
def test_pearce_hall_standard_update(kappa, gamma):
    """Test standard update with various learning rates."""
    value = jnp.array(0.5)
    alpha = jnp.array(0.5)
    outcome = jnp.array(1.0)
    chosen = jnp.array(1.0)
    outcome_chosen = (outcome, chosen)
    
    # Run Function
    (new_value, new_alpha), (old_val, old_alpha, pe) = pearce_hall_update(
        value, alpha, outcome_chosen, kappa, gamma
    )
    
    # Calculate Expected
    exp_val, exp_alpha, exp_pe = expected_ph_update(0.5, 0.5, 1.0, 1.0, kappa, gamma)
    
    assert np.isclose(pe, exp_pe)
    assert np.isclose(new_value, exp_val)
    assert np.isclose(new_alpha, exp_alpha)

def test_pearce_hall_update_all_options():
    """Test that update_all_options=True updates unchosen options."""
    value = jnp.array(0.5)
    alpha = jnp.array(0.5)
    outcome = jnp.array(1.0)
    chosen = jnp.array(0.0) # Unchosen!
    outcome_chosen = (outcome, chosen)
    kappa = jnp.array(0.5)
    gamma = jnp.array(0.5)
    
    # With update_all_options=True, we treat 'chosen' mask logic differently inside? 
    # Actually, your implementation uses: 
    # chosen = (chosen * (1 - update_all_options)) + (1 * update_all_options)
    # So effective_chosen becomes 1.
    
    (new_value, new_alpha), (_, _, pe) = pearce_hall_update(
        value, alpha, outcome_chosen, kappa, gamma, update_all_options=True
    )
    
    # If effective chosen is 1, PE is calculated:
    # PE = 1.0 - 0.5 = 0.5
    # Val = 0.5 + 0.5*0.5*0.5 = 0.625
    
    assert np.isclose(pe, 0.5) 
    assert np.isclose(new_value, 0.625) 
    # Alpha should also update
    # alpha_new = 0.5 + 1.0 * (0.5 * (0.5 - 0.5)) = 0.5 (PE matches alpha here accidentally)
    
    # Try different numbers to verify alpha update
    # Alpha=0.8, PE=0.5 -> |PE| < Alpha -> Alpha decreases
    alpha_high = jnp.array(0.8)
    (new_value_2, new_alpha_2), _ = pearce_hall_update(
        value, alpha_high, outcome_chosen, kappa, gamma, update_all_options=True
    )
    
    # Exp alpha: 0.8 + 0.5 * (|0.5| - 0.8) = 0.8 + 0.5 * (-0.3) = 0.8 - 0.15 = 0.65
    assert np.isclose(new_alpha_2, 0.65)

def test_custom_counterfactual():
    """Test that limits updates based on custom counterfactual logic."""
    value = jnp.array(0.5)
    alpha = jnp.array(0.5)
    outcome = jnp.array(1.0)
    chosen = jnp.array(0.0)
    outcome_chosen = (outcome, chosen)
    
    # Custom: unchosen options get reward -1.0
    def custom_cf(r, c): return -1.0
    
    # With standard unchosen logic (update_all=False), PE is 0 anyway.
    # But checking internal logic might require introspection or update_all=True
    # If update_all=True, it uses the counterfactual as the outcome.
    
    (new_value, _), (_, _, pe) = pearce_hall_update(
        value, alpha, outcome_chosen, 0.5, 0.5, 
        counterfactual_value=custom_cf, update_all_options=True
    )
    
    # Outcome becomes -1.0. PE = -1.0 - 0.5 = -1.5
    assert np.isclose(pe, -1.5)
    
@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_shapes_and_broadcasting(shape):
    """Test that arrays of various shapes work (vectorization)."""
    value = jnp.ones(shape) * 0.5
    alpha = jnp.ones(shape) * 0.5
    outcome = jnp.ones(shape)
    chosen = jnp.ones(shape)
    
    (new_val, new_alpha), _ = pearce_hall_update(
        value, alpha, (outcome, chosen), 0.5, 0.5
    )
    
    assert new_val.shape == shape
    assert new_alpha.shape == shape

def test_pearce_hall_jit():
    """Verify JIT compilation works."""
    update_jit = jax.jit(pearce_hall_update)
    # Simple smoke test
    update_jit(
        jnp.array(0.5), jnp.array(0.5), 
        (jnp.array(1.0), jnp.array(1.0)), 
        0.5, 0.5
    )
