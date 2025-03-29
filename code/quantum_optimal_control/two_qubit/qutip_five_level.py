import numpy as np
from qutip import basis, tensor, mesolve, Options

def gen_n_level_atom_basis(n: int):
    """
    Generate a list of QuTiP basis states: [|0>, |1>, ..., |n-1>]
    for an n-dimensional Hilbert space.
    """
    return [basis(n, i) for i in range(n)]

def QTHamiltonian(argsME5lvl):
    """
    Construct the time-dependent Hamiltonian for a 5-level two-qubit system.

    Args:
        argsME5lvl: A list containing:
            [
              psi0, psi_orth, V_int, Rabi_i_max, Rabi_r_max,
              Rabi_i_Pulse_Re, Rabi_r_Pulse_Re, Rabi_i_Pulse_Im,
              Rabi_r_Pulse_Im, Delta_i_Pulse, Delta_i_max,
              del_total, Gammas
            ]
            where each element is as described in the original notebook code.
    Returns:
        A list that QuTiP interprets as a time-dependent Hamiltonian:
        [H_0, [H_1, f_1(t)], [H_2, f_2(t)], ...]
    """
    psi0, psi_orth, V_int, Rabi_i_max, Rabi_r_max, Rabi_i_Pulse_Re, Rabi_r_Pulse_Re, Rabi_i_Pulse_Im, \
        Rabi_r_Pulse_Im, Delta_i_Pulse, Delta_i_max, del_total, Gammas = argsME5lvl

    dim = 5
    # Single-qubit 5-level basis states
    g_0, g_1, i, r, dark = gen_n_level_atom_basis(dim)

    # Projectors
    sig_rr = r * r.dag()
    sig_ii = i * i.dag()

    # Single-qubit Hamiltonians
    # 1) delta detuning for Rydberg
    H_SQ_del = - del_total * sig_rr
    # 2) single-photon detuning
    H_SQ_Delta_i = - Delta_i_max * sig_ii

    # Coupling operators for Rabi frequencies (real & imaginary parts)
    sig_1i = g_1 * i.dag()
    sig_ir = i * r.dag()

    H_SQ_Rabi_i_re = -0.5 * Rabi_i_max * (sig_1i + sig_1i.dag())
    H_SQ_Rabi_i_im = -0.5j * Rabi_i_max * (sig_1i - sig_1i.dag())
    H_SQ_Rabi_r_re = -0.5 * Rabi_r_max * (sig_ir + sig_ir.dag())
    H_SQ_Rabi_r_im = -0.5j * Rabi_r_max * (sig_ir - sig_ir.dag())

    # Two-qubit dimension expansions
    from qutip import identity

    H_TQ_del_total = tensor(identity(dim), H_SQ_del) + tensor(H_SQ_del, identity(dim))
    # Rydberg-Rydberg interaction
    H_TQ_0 = H_TQ_del_total + V_int * tensor(sig_rr, sig_rr)

    H_TQ_Delta = tensor(identity(dim), H_SQ_Delta_i) + tensor(H_SQ_Delta_i, identity(dim))

    H_TQ_Rabi_i_re = tensor(H_SQ_Rabi_i_re, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_i_re)
    H_TQ_Rabi_i_im = tensor(H_SQ_Rabi_i_im, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_i_im)

    H_TQ_Rabi_r_re = tensor(H_SQ_Rabi_r_re, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_r_re)
    H_TQ_Rabi_r_im = tensor(H_SQ_Rabi_r_im, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_r_im)

    # Build time-dependent Hamiltonian list
    H = [
        H_TQ_0,
        [H_TQ_Delta, Delta_i_Pulse],
        [H_TQ_Rabi_i_re, Rabi_i_Pulse_Re],
        [H_TQ_Rabi_i_im, Rabi_i_Pulse_Im],
        [H_TQ_Rabi_r_re, Rabi_r_Pulse_Re],
        [H_TQ_Rabi_r_im, Rabi_r_Pulse_Im],
    ]
    return H

def Mesolve_5lvl_t(time, argsME5lvl, output_states=False, options=None):
    """
    Perform time evolution using the 5-level model for two qubits with decay.

    Args:
        time: The time array for evolution (1D numpy array).
        argsME5lvl: Same structure as used by QTHamiltonian.
        output_states: Whether to track expectation values or full states.
        options: QuTiP solver options, can be None.

    Returns:
        A Qutip Result object from the mesolve routine.
    """
    from qutip import identity, tensor, Options

    psi0, psi_orth, V_int, Rabi_i_max, Rabi_r_max, Rabi_i_Pulse_Re, Rabi_r_Pulse_Re, Rabi_i_Pulse_Im, \
        Rabi_r_Pulse_Im, Delta_i_Pulse, Delta_i_max, del_total, Gammas = argsME5lvl

    # Build the Hamiltonian
    H = QTHamiltonian(argsME5lvl)

    dim = 5
    g_0, g_1, i, r, dark = gen_n_level_atom_basis(dim)

    sig_01 = g_0 * g_1.dag()
    sig_1i = g_1 * i.dag()
    sig_ir = i * r.dag()
    sig_dr = dark * r.dag()

    # Build the collapse operators for decay
    # Gammas_list = [Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd]
    Decay_ops = []
    ops = [sig_01, sig_1i, sig_ir, sig_dr]

    for Gamma_ind in range(len(ops)):
        for qubit_ind in range(2):
            # Each operator goes on either qubit 1 or qubit 2
            op_list = [identity(dim), identity(dim)]
            op_list[qubit_ind] = ops[Gamma_ind]
            Decay_ops.append(np.sqrt(Gammas[Gamma_ind]) * tensor(op_list))

    # Default solver options if none provided
    if options is None:
        options1 = Options(max_step=1e-9, nsteps=1000000000)
    else:
        options1 = options

    if not output_states:
        observables = []
    else:
        # Example: track fidelity or population, etc.
        from qutip import expect
        observables = [psi0 * psi0.dag(), psi_orth * psi_orth.dag()]

    result = mesolve(
        H,
        psi0,
        time,
        Decay_ops,
        observables,
        options=options1,
        args={}
    )
    return result