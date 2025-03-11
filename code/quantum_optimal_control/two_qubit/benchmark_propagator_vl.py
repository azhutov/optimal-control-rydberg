#!/usr/bin/env python3
"""
benchmark_propagator_vl.py

This script profiles a short run of the quantum optimal control code
using Python's built-in cProfile. It sets up the PropagatorVL instance,
performs a few gradient-based optimization steps, and prints out the
most time-consuming functions. Use this to detect bottlenecks.

Usage:
    python benchmark_propagator_vl.py
"""

import cProfile
import pstats
import io
import numpy as np
import tensorflow as tf
import time

from arc import Rubidium87
from qutip import *
from quantum_optimal_control.two_qubit.propagator_vl import PropagatorVL

def create_propagator_and_optimizer():
    """
    Sets up the PropagatorVL instance and Adam optimizer with a small system.
    Returns:
        propagator (PropagatorVL) - instance of the quantum control propagator
        optimizer (tf.keras.optimizers.Optimizer) - the optimizer for gradient steps
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
    propagator = PropagatorVL(
        input_dim, nt, pad, f_std, delta_t, del_total, V_int,
        Delta_i, Rabi_i, Rabi_r, Gammas
    )

    # Randomly assign initial Gaussian parameters
    numb_ctrl_amps = 5
    propagator.ctrl_amplitudes_a.assign(
        tf.random.uniform([input_dim, numb_ctrl_amps], -1, 1, dtype=tf.float64)
    )
    propagator.ctrl_amplitudes_b.assign(
        tf.random.uniform([input_dim, numb_ctrl_amps], -1, 1, dtype=tf.float64)
    )
    propagator.ctrl_amplitudes_c.assign(
        tf.random.uniform([input_dim, numb_ctrl_amps], 0, 0.5, dtype=tf.float64)
    )

    # Create optimizer
    learning_rate = 0.02
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    return propagator, optimizer, tlist

@tf.function
def single_optimization_step(propagator, optimizer):
    """
    Perform one gradient-based optimization iteration.
    """
    with tf.GradientTape() as tape:
        cost = propagator.target()
    grads = tape.gradient(cost, [
        propagator.ctrl_amplitudes_a,
        propagator.ctrl_amplitudes_b,
        propagator.ctrl_amplitudes_c,
    ])
    optimizer.apply_gradients(zip(
        grads, [
            propagator.ctrl_amplitudes_a,
            propagator.ctrl_amplitudes_b,
            propagator.ctrl_amplitudes_c
        ]
    ))
    return cost

def benchmark_propagation(num_steps=10):
    """
    Creates the propagator, then runs 'num_steps' optimization steps.
    Tracks the final cost, to ensure code is meaningfully running.
    """
    propagator, optimizer, _ = create_propagator_and_optimizer()

    best_cost = np.inf
    for step in range(num_steps):
        current_cost = single_optimization_step(propagator, optimizer)
        cost_val = current_cost.numpy()
        if cost_val < best_cost:
            best_cost = cost_val
    return best_cost

def main():
    """
    Profiles the benchmark_propagation function using cProfile,
    then prints out the most time-consuming functions.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the profiling target
    final_cost = benchmark_propagation(num_steps=20)

    profiler.disable()

    print(f"Final cost after 20 steps: {final_cost[0][0]:.6e}")

    # Print out profiling stats
    s = io.StringIO()
    sortby = 'tottime'   # or 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(30)   # print top 30 lines
    print(s.getvalue())

if __name__ == "__main__":
    # Suppress TF warnings to see clearer profiling output
    tf.get_logger().setLevel('ERROR')

    main()