from qutip import basis, tensor, identity, mesolve, Options
import numpy as np

def gen_n_level_atom_basis(n):
    return [basis(n, n_l) for n_l in range(0, n)]

def QTHamiltonian(argsME5lvl):
    psi0, psi_orth, V_int, Rabi_i_max, Rabi_r_max, Rabi_i_Pulse_Re, Rabi_r_Pulse_Re, Rabi_i_Pulse_Im, Rabi_r_Pulse_Im, Delta_i_Pulse, Delta_i_max, del_total, Gammas = argsME5lvl

    # set dimension of Hilbert space:
    dim = 5

    # Set operators:
    g_0, g_1, i, r, dark = gen_n_level_atom_basis(5)

    sig_00 = g_0 * g_0.dag()
    sig_11 = g_1 * g_1.dag()
    sig_ii = i * i.dag()
    sig_rr = r * r.dag()
    sig_01 = g_0 * g_1.dag()
    sig_1i = g_1 * i.dag()
    sig_dr = dark * r.dag()
    sig_ir = i * r.dag()

    # Set single qubit hamiltonian parts:
    H_SQ_del = - del_total * sig_rr
    H_SQ_Delta_i = - Delta_i_max * sig_ii
    H_SQ_Rabi_i_re = - 1/2 * Rabi_i_max * (sig_1i + sig_1i.dag())
    H_SQ_Rabi_i_im = - 1j/2 * Rabi_i_max * (sig_1i - sig_1i.dag())
    H_SQ_Rabi_r_re = - 1/2 * Rabi_r_max * (sig_ir + sig_ir.dag())
    H_SQ_Rabi_r_im = - 1j/2 * Rabi_r_max * (sig_ir - sig_ir.dag())

    # Set two qubit static terms
    H_TQ_del_total = tensor(identity(dim), H_SQ_del) + tensor(H_SQ_del, identity(dim))
    H_TQ_0 = H_TQ_del_total + V_int * tensor(sig_rr, sig_rr)

    # Set single photon detunings:
    H_TQ_Delta = tensor(identity(dim), H_SQ_Delta_i) + tensor(H_SQ_Delta_i, identity(dim))

    # Real part
    # Set drive 1:(use same power on each qubit))
    H_TQ_Rabi_i_re = tensor(H_SQ_Rabi_i_re, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_i_re)
    # Set second qubit drive 1:
    H_TQ_Rabi_r_re = tensor(H_SQ_Rabi_r_re, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_r_re)

    # Imaginary part
    H_TQ_Rabi_i_im = tensor(H_SQ_Rabi_i_im, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_i_im)
    # Set second qubit drive 1:
    H_TQ_Rabi_r_im = tensor(H_SQ_Rabi_r_im, identity(dim)) + tensor(identity(dim), H_SQ_Rabi_r_im)

    H = [H_TQ_0, [H_TQ_Delta, Delta_i_Pulse], [H_TQ_Rabi_i_re, Rabi_i_Pulse_Re],
         [H_TQ_Rabi_i_im, Rabi_i_Pulse_Im], [H_TQ_Rabi_r_re, Rabi_r_Pulse_Re],
         [H_TQ_Rabi_r_im, Rabi_r_Pulse_Im]
         ]
    return H

def Mesolve_5lvl_t(time, argsME5lvl, output_states=False, options=None):
    psi0, psi_orth, V_int, Rabi_i_max, Rabi_r_max, Rabi_i_Pulse_Re, Rabi_r_Pulse_Re, Rabi_i_Pulse_Im, Rabi_r_Pulse_Im, Delta_i_Pulse, Delta_i_max, del_total, Gammas = argsME5lvl

    H = QTHamiltonian(argsME5lvl)

    # set dimension of Hilbert space:
    dim = 5

    # Set operators:
    g_0, g_1, i, r, dark = gen_n_level_atom_basis(dim)

    sig_00 = g_0 * g_0.dag()
    sig_11 = g_1 * g_1.dag()
    sig_ii = i * i.dag()
    sig_rr = r * r.dag()
    sig_01 = g_0 * g_1.dag()
    sig_1i = g_1 * i.dag()
    sig_dr = dark * r.dag()
    sig_ir = i * r.dag()

    # Set decay:
    Decay_ops = []
    ops = [sig_01, sig_1i, sig_ir, sig_dr]
    Gammas_list = Gammas

    # Loop over each Gamma:
    for Gamma_ind in range(len(ops)):
        # loop for qubit 1 and 2:
        for qubit_ind in range(2):
            op = [identity(dim), identity(dim)]
            op[qubit_ind] = ops[Gamma_ind]
            Decay = np.sqrt(Gammas_list[Gamma_ind]) * tensor(op)
            Decay_ops.append(Decay)

    if options is None:
        options1 = Options(max_step=1e-9, nsteps=1e9)
    else:
        options1 = options

    if not output_states:
        observables = []
    else:
        observables = [psi0 * psi0.dag(), psi_orth * psi_orth.dag(),
                       tensor(sig_ii, identity(dim)) * tensor(sig_ii, identity(dim)).dag(),
                       tensor(sig_rr, sig_rr) * tensor(sig_rr, sig_rr).dag()
                       ]
    result = mesolve(H, psi0, time, Decay_ops, observables, options=options1, args={})
    return result