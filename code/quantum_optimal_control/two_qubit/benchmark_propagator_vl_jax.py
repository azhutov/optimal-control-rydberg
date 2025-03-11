#!/usr/bin/env python3
"""
benchmark_propagator_vl_jax.py

This script profiles a short run of the JAX-based quantum optimal control code
using Python's built-in cProfile. It sets up the PropagatorVLJAX instance,
performs a few gradient-based optimization steps, and prints out the
most time-consuming functions. Use this to detect bottlenecks.

Usage:
    python benchmark_propagator_vl_jax.py
"""

import cProfile
import pstats
import io
import numpy as np
import jax
import jax.numpy as jnp
import time

from arc import Rubidium87
from qutip import *
from quantum_optimal_control.two_qubit.propagator_vl_jax import PropagatorVLJAX, single_optimization_step

# Set JAX to use GPU if available
try:
    jax.config.update('jax_platform_name', 'gpu')
    print(f"JAX is using: {jax.devices()[0]}")
except:
    print("No GPU found for JAX, using CPU.")


def create_propagator():
    """
    Sets up the PropagatorVLJAX instance with a small system.
    Returns:
        propagator (PropagatorVLJAX) - instance of the JAX quantum control propagator
        ctrl_a, ctrl_b, ctrl_c (jnp.ndarray) - initial control parameters
        tlist (np.ndarray) - time array (for potential plotting or verification)
    """
    # --- System parameters (similar to the notebook) ---
    atom = Rubidium87()

    n_i = 6
    l_i = 1
    j_i = 1.5
    T_i = atom.getStateLifetime(n_i, l_i, j_i)
    Gamma_ig = 1/T_i

    # Additional decay channels for two-qubit gate
    Gamma_i1 = 1 * Gamma_ig  # Allowed decay
    Gamma_i0 = 0            # Not used, but for reference
    Gamma_10 = 0.1          # Hyperfine ground state decay rate

    # Rydberg states
    n_r = 70
    l_r = 0
    j_r = 0.5
    T_rTot = atom.getStateLifetime(n_r, l_r, j_r, temperature=300, includeLevelsUpTo=n_r + 50)
    T_rRad = atom.getStateLifetime(n_r, l_r, j_r, temperature=0)
    T_ri = 1/atom.getTransitionRate(n_r, l_r, j_r, n_i, l_i, j_i, temperature=0)

    T_rgp = 1/(1/T_rRad - 1/T_ri)
    T_rBB = 1/(1/T_rTot - 1/T_rRad)

    Gamma_ri = 1/T_ri
    Gamma_rrp = 1/T_rBB
    Gamma_rgp = 1/T_rgp
    Gamma_rTot = Gamma_ri + Gamma_rrp + Gamma_rgp
    Gamma_rd = Gamma_rrp + Gamma_rgp

    Gammas = [Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd]

    # Two-qubit gate parameters
    V_int = 2 * np.pi * 10e6
    tau = 324e-9
    Delta_i = 2 * np.pi * -35.7e6

    Rabi_i = 2 * np.pi * 100e6
    Rabi_r = 2 * np.pi * 100e6
    del_total = 0

    # Time grid
    t_0 = 0
    t_f = 2 * tau
    nt = 200  # smaller number for quicker profiling
    pad = int(0.03 * nt)
    delta_t = (t_f - t_0) / (nt + 2 * pad)
    tlist = np.linspace(t_0, t_f, nt + 2 * pad)
    f_std = 50e6
    input_dim = 10

    # Create the propagator
    propagator = PropagatorVLJAX(
        input_dim, nt, pad, f_std, delta_t, del_total, V_int,
        Delta_i, Rabi_i, Rabi_r, Gammas
    )

    # Randomly assign initial Gaussian parameters
    numb_ctrl_amps = 5
    # Use JAX random number generator with a key
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    
    ctrl_a = jax.random.uniform(subkey1, (input_dim, numb_ctrl_amps), 
                                minval=-1.0, maxval=1.0, dtype=jnp.float64)
    ctrl_b = jax.random.uniform(subkey2, (input_dim, numb_ctrl_amps), 
                                minval=-1.0, maxval=1.0, dtype=jnp.float64)
    ctrl_c = jax.random.uniform(subkey3, (input_dim, numb_ctrl_amps), 
                                minval=0.0, maxval=0.5, dtype=jnp.float64)

    return propagator, ctrl_a, ctrl_b, ctrl_c, tlist


def benchmark_propagation(num_steps=10):
    """
    Creates the propagator, then runs 'num_steps' optimization steps.
    Tracks the final cost and time per iteration.
    """
    propagator, ctrl_a, ctrl_b, ctrl_c, _ = create_propagator()
    learning_rate = 0.02

    # In JAX, we need to track the state manually between iterations
    best_cost = np.inf
    times = []
    costs = []
    
    # Pre-compile the function (first run will be slower due to JIT)
    cost, ctrl_a, ctrl_b, ctrl_c = single_optimization_step(
        propagator, ctrl_a, ctrl_b, ctrl_c, learning_rate
    )
    
    for step in range(num_steps):
        start_time = time.time()
        cost, ctrl_a, ctrl_b, ctrl_c = single_optimization_step(
            propagator, ctrl_a, ctrl_b, ctrl_c, learning_rate
        )
        end_time = time.time()
        
        # JAX arrays are immutable, so we need to convert to numpy to use them
        cost_val = np.array(cost)
        if cost_val < best_cost:
            best_cost = cost_val
        times.append(end_time - start_time)
        costs.append(cost_val)
    
    with open('cost_vs_iteration.txt', 'w') as f:
        for i, c in enumerate(costs):
            f.write(f"{i} {c:.6f}\n")

    return best_cost, times


def main():
    """
    Profiles the benchmark_propagation function using cProfile,
    then prints out the most time-consuming functions.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    num_steps = 50
    # Run the profiling target
    final_cost, times = benchmark_propagation(num_steps=num_steps)

    profiler.disable()

    print(f"Final cost after {num_steps} steps: {final_cost:.6e}")
    print(f"Average time per iteration: {np.mean(times):.6f} s")

    # Print out profiling stats
    s = io.StringIO()
    sortby = 'tottime'   # or 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(30)   # print top 30 lines
    print(s.getvalue())

    # Write results to file
    with open('benchmark_propagation_jax_gpu.txt', 'w') as f:
        f.write(f"Final cost after {num_steps} steps: {final_cost:.6e}\n")
        f.write(f"Average time per iteration: {np.mean(times):.6f} s\n")
        f.write(f"Iteration times: {times}\n")
        f.write(s.getvalue())


if __name__ == "__main__":
    main() 