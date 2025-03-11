import numpy as np
import scipy.signal
import tensorflow as tf
from qutip import *

class PropagatorVL:
    def __init__(
        self, input_dim, no_of_steps, pad, f_std, delta_t, del_total, V_int,
        Delta_i, Rabi_i, Rabi_r, Gammas_all
    ):
        """
        The constructor sets up all needed states, operators, and internal references.
        We avoid repeated allocations by caching constants.
        """
        self.duration = (no_of_steps + 2 * pad) * delta_t
        self.delta_t = delta_t
        self.Vint = V_int
        self.Rabi_i = Rabi_i  # Max i Rabi
        self.Rabi_r = Rabi_r  # Max r Rabi
        self.Delta_i = Delta_i
        self.del_total = del_total
        self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd = Gammas_all
        self.dim = 4  # Dimension for each atom (4-level system)
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps
        self.padding = pad

        # Create a time axis in float64 so we do not mix float32/float64
        self.times_basis = tf.linspace(
            tf.constant(-1.0, dtype=tf.float64),
            tf.constant(1.0, dtype=tf.float64),
            no_of_steps,
            name="basis_linspace"
        )

        # Generate the Gaussian filter in frequency domain once
        # shape => (no_of_steps + 2*pad)
        freqs = np.fft.fftfreq(no_of_steps, d=delta_t)
        std_filter = f_std / (freqs[1] - freqs[0])
        gaussian_array = scipy.signal.gaussian(no_of_steps + 2 * pad, std_filter)
        self.Gaussian_filter = tf.constant(gaussian_array, dtype=tf.complex128)

        # Build single-atom basis states
        g_0, g_1, i, r = self.nLevelAtomBasis(self.dim)

        # Build initial states and target operator
        self.psi_pp = tf.cast(
            0.5 * (
                tensor(g_0, g_0) + tensor(g_0, g_1)
                + tensor(g_1, g_0) + tensor(g_1, g_1)
            ),
            dtype=tf.complex128
        )
        self.psi_10 = tf.cast(tensor(g_1, g_0), dtype=tf.complex128)
        self.psi_11 = tf.cast(tensor(g_1, g_1), dtype=tf.complex128)
        self.psi_01 = tf.cast(tensor(g_0, g_1), dtype=tf.complex128)
        self.U_target = (
            tensor(g_0, g_0) * tensor(g_0, g_0).dag()
            + tensor(g_0, g_1) * tensor(g_0, g_1).dag()
            + tensor(g_1, g_0) * tensor(g_1, g_0).dag()
            - tensor(g_1, g_1) * tensor(g_1, g_1).dag()
        )

        # Build the Van Loan generator blocks
        Ham_args = [
            g_0, g_1, i, r,
            self.dim, self.del_total, self.Vint,
            self.Delta_i, self.Rabi_i, self.Rabi_r,
            self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd
        ]
        basis = [g_0, g_1, i, r]
        self.VL_Generators_ansatz = self.VL_Generators(
            self.Hamiltonian(Ham_args), basis, self.dim
        )

        # Control amplitude variables (a,b,c). shape => [input_dim, num_ctrls]
        self.num_ctrls = 5
        self.ctrl_amplitudes_a = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64),
            dtype=tf.float64
        )
        self.ctrl_amplitudes_b = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64),
            dtype=tf.float64
        )
        self.ctrl_amplitudes_c = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64),
            dtype=tf.float64
        )
        self.ctrl_amplitudes_all = tf.concat(
            [self.ctrl_amplitudes_a, self.ctrl_amplitudes_b, self.ctrl_amplitudes_c],
            axis=1
        )

    @staticmethod
    def nLevelAtomBasis(n):
        states = []
        for n_l in range(n):
            states.append(qutip.basis(n, n_l))
        return states

    @staticmethod
    def Hamiltonian(args):
        """
        Ladder-system Hamiltonian, including non-Hermitian decay and interactions.
        """
        (
            g_0, g_1, i, r,
            dim, del_total, V_int,
            Delta_i, Rabi_i, Rabi_r,
            Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd
        ) = args

        sig_00 = g_0 * g_0.dag()
        sig_11 = g_1 * g_1.dag()
        sig_ii = i * i.dag()
        sig_rr = r * r.dag()
        sig_1i = g_1 * i.dag()
        sig_ir = i * r.dag()

        # Effective single-atom decay
        H_SQ_decay = (
            -1j * Gamma_i1 / 2.0 * sig_ii
            -1j * Gamma_10 / 2.0 * sig_11
            -1j * (Gamma_ri + Gamma_rd) / 2.0 * sig_rr
        )
        H_SQ_del = - del_total * sig_rr
        H_SQ_Delta_i = - Delta_i * sig_ii

        # Rabi couplings (real + imaginary parts)
        H_SQ_Rabi_i_re = -0.5 * Rabi_i * (sig_1i + sig_1i.dag())
        H_SQ_Rabi_i_im = -0.5j * Rabi_i * (sig_1i - sig_1i.dag())
        H_SQ_Rabi_r_re = -0.5 * Rabi_r * (sig_ir + sig_ir.dag())
        H_SQ_Rabi_r_im = -0.5j * Rabi_r * (sig_ir - sig_ir.dag())

        # Two-qubit operators
        H_TQ_0 = (
            tensor(identity(dim), H_SQ_decay)
            + tensor(H_SQ_decay, identity(dim))
            + V_int * tensor(sig_rr, sig_rr)
        )
        H_TQ_del_total = tensor(identity(dim), H_SQ_del) + tensor(H_SQ_del, identity(dim))
        H_TQ_Delta = tensor(identity(dim), H_SQ_Delta_i) + tensor(H_SQ_Delta_i, identity(dim))
        H_TQ_Rabi_i_re = (
            tensor(H_SQ_Rabi_i_re, identity(dim))
            + tensor(identity(dim), H_SQ_Rabi_i_re)
        )
        H_TQ_Rabi_i_im = (
            tensor(H_SQ_Rabi_i_im, identity(dim))
            + tensor(identity(dim), H_SQ_Rabi_i_im)
        )
        H_TQ_Rabi_r_re = (
            tensor(H_SQ_Rabi_r_re, identity(dim))
            + tensor(identity(dim), H_SQ_Rabi_r_re)
        )
        H_TQ_Rabi_r_im = (
            tensor(H_SQ_Rabi_r_im, identity(dim))
            + tensor(identity(dim), H_SQ_Rabi_r_im)
        )

        return [
            H_TQ_0,           # 0
            H_TQ_del_total,   # 1
            H_TQ_Delta,       # 2
            H_TQ_Rabi_i_re,   # 3
            H_TQ_Rabi_i_im,   # 4
            H_TQ_Rabi_r_re,   # 5
            H_TQ_Rabi_r_im    # 6
        ]

    @staticmethod
    def VL_Generators(Hamiltonian, basis, dim):
        """
        Construct block-lifted operators for Van Loan approach.
        Dimension => 2*(dim^2).
        """
        U_Decay = Hamiltonian[0]
        U1 = Hamiltonian[1]
        U2 = Hamiltonian[2]
        U3_real = Hamiltonian[3]
        U3_imag = Hamiltonian[4]
        U4_real = Hamiltonian[5]
        U4_imag = Hamiltonian[6]

        Zeroes = np.zeros((dim**2, dim**2), dtype=np.complex128)

        LU_decay = np.block([
            [U_Decay, Zeroes],
            [Zeroes, U_Decay]
        ])
        LU1 = np.block([
            [U1, Zeroes],
            [Zeroes, U1]
        ])
        LU2 = np.block([
            [U2, Zeroes],
            [Zeroes, U2]
        ])
        LU3_real = np.block([
            [U3_real, Zeroes],
            [Zeroes, U3_real]
        ])
        LU3_imag = np.block([
            [U3_imag, Zeroes],
            [Zeroes, U3_imag]
        ])
        LU4_real = np.block([
            [U4_real, Zeroes],
            [Zeroes, U4_real]
        ])
        LU4_imag = np.block([
            [U4_imag, Zeroes],
            [Zeroes, U4_imag]
        ])

        # Not used, but we keep them consistent
        g_0, g_1, i, r = basis
        LP1 = np.block([
            [Zeroes, tensor(g_1*g_1.dag(), g_0*g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP2 = np.block([
            [Zeroes, tensor(r*r.dag(), g_0*g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP3 = np.block([
            [Zeroes, tensor(g_1*r.dag(), g_0*g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP4 = np.block([
            [Zeroes, tensor(r*g_1.dag(), g_0*g_0.dag())],
            [Zeroes, Zeroes]
        ])

        return tf.stack([
            tf.constant(LU_decay, dtype=tf.complex128),
            tf.constant(LU1, dtype=tf.complex128),
            tf.constant(LU2, dtype=tf.complex128),
            tf.constant(LU3_real, dtype=tf.complex128),
            tf.constant(LU3_imag, dtype=tf.complex128),
            tf.constant(LU4_real, dtype=tf.complex128),
            tf.constant(LU4_imag, dtype=tf.complex128),
            tf.constant(LP1, dtype=tf.complex128),
            tf.constant(LP2, dtype=tf.complex128),
            tf.constant(LP3, dtype=tf.complex128),
            tf.constant(LP4, dtype=tf.complex128)
        ])

    def gen_basis_matrix(self, amp_b, amp_c):
        """
        Creates Gaussian modes over self.times_basis by exp(-((t - b)^2 / c^2)).
        We keep everything float64.
        """
        def single_mode(params):
            b_val, c_val = params[0], params[1]  # both float64
            # Gaussian in time
            return tf.exp(-((self.times_basis - b_val)**2) / (c_val**2))

        combined = tf.stack([amp_b, amp_c], axis=1)  # [input_dim, 2]
        # map_fn => shape [input_dim, no_of_steps]
        Gaussian_modes = tf.map_fn(single_mode, combined, dtype=tf.float64)
        # Transpose => [no_of_steps, input_dim]
        return tf.transpose(Gaussian_modes)

    def transform_amplitudes(self):
        """
        Build time-domain waveforms from the a,b,c parameters using Gaussian basis.
        """
        # clamp b,c to [-1,1] in float64
        b_clamped = tf.math.tanh(self.ctrl_amplitudes_b)  # float64
        c_clamped = tf.math.tanh(self.ctrl_amplitudes_c)  # float64

        all_outputs = []
        for control_num in range(self.num_ctrls):
            basis_mat = self.gen_basis_matrix(
                b_clamped[:, control_num], c_clamped[:, control_num]
            )  # [no_of_steps, input_dim]
            amplitude_col = self.ctrl_amplitudes_a[:, control_num : control_num+1]  # [input_dim,1]
            wave = tf.linalg.matmul(basis_mat, amplitude_col)  # => [no_of_steps,1]
            all_outputs.append(wave)
        return tf.concat(all_outputs, axis=1)  # [no_of_steps, 5]

    def regularize_amplitudes(self):
        """
        1) Create basis waveforms -> transform into physically-bounded waveforms
        2) The columns are: (Delta, Rabi_i amplitude, Rabi_i phase, Rabi_r amplitude, Rabi_r phase)
        """
        ta = self.transform_amplitudes()  # [no_of_steps, 5]

        # Delta => always -1 in original logic
        Delta_col = 0.0 * ta[:, 0:1] - 1.0

        # Rabi i amplitude => 1 - exp(-ta^2)
        Rabi_i_mag = 1.0 - tf.exp(-tf.square(ta[:, 1:2]))
        Rabi_i_phase = tf.math.tanh(ta[:, 2:3])

        Rabi_r_mag = 1.0 - tf.exp(-tf.square(ta[:, 3:4]))
        Rabi_r_phase = tf.math.tanh(ta[:, 4:5])

        return tf.concat(
            [Delta_col, Rabi_i_mag, Rabi_i_phase, Rabi_r_mag, Rabi_r_phase],
            axis=1
        )

    @staticmethod
    def filter_amplitudes(amplitudes, filter_function_array, padding):
        """
        Apply frequency-domain Gaussian filter to each control channel.
        amplitudes => [no_of_steps, 5], pad => no_of_steps+2*padding
        """
        padded = tf.pad(
            tf.cast(amplitudes, dtype=tf.complex128),
            [[padding, padding], [0, 0]],
            "CONSTANT"
        )

        def fft_filter_ifft(column):
            freq_amp = tf.signal.fftshift(tf.signal.fft(column))
            filtered = tf.multiply(filter_function_array, freq_amp)
            return tf.signal.ifft(tf.signal.ifftshift(filtered))

        result_cols = []
        for col_i in range(5):
            filtered_col = fft_filter_ifft(padded[:, col_i])
            result_cols.append(tf.cast(filtered_col, dtype=tf.float64))

        filtered_all = tf.stack(result_cols, axis=1)
        return filtered_all

    def return_physical_amplitudes(self):
        """
        1) regularize_amplitudes -> wave
        2) frequency filter -> final shape [N + 2*pad, 5]
        """
        reg = self.regularize_amplitudes()
        filtered = self.filter_amplitudes(reg, self.Gaussian_filter, self.padding)
        return filtered

    def return_auxiliary_amplitudes(self):
        """
        Convert final filtered amplitude array to complex controls:
         - wave[:,0] => Delta_i
         - wave[:,1], wave[:,2] => Rabi_i (amplitude, phase)
         - wave[:,3], wave[:,4] => Rabi_r (amplitude, phase)
        Also build P projection terms (LP1..).
        Return => shape [N + 2*pad, 9], first 5 columns = U, last 4 = P.
        """
        wave = self.return_physical_amplitudes()  # [N, 5], N=no_of_steps+2*pad

        # build complex Rabi for i
        rabi_i_mag = self.Rabi_i * wave[:, 1:2]  # float64
        rabi_i_phase = np.pi * wave[:, 2:3]
        rabi_i_cplx = tf.complex(
            rabi_i_mag * tf.math.cos(rabi_i_phase),
            rabi_i_mag * tf.math.sin(rabi_i_phase)
        )

        # build complex Rabi for r
        rabi_r_mag = self.Rabi_r * wave[:, 3:4]
        rabi_r_phase = np.pi * wave[:, 4:5]
        rabi_r_cplx = tf.complex(
            rabi_r_mag * tf.math.cos(rabi_r_phase),
            rabi_r_mag * tf.math.sin(rabi_r_phase)
        )

        # P projection amplitudes
        LP_1 = tf.multiply(rabi_r_cplx, tf.math.conj(rabi_r_cplx))  # [N,1]
        LP_2 = tf.multiply(rabi_i_cplx, tf.math.conj(rabi_i_cplx))  # [N,1]
        LP_3 = - tf.multiply(rabi_i_cplx, rabi_r_cplx)
        LP_4 = - tf.multiply(tf.math.conj(rabi_i_cplx), tf.math.conj(rabi_r_cplx))

        norm_denom = LP_1 + LP_2
        P_stacked = tf.concat([LP_1, LP_2, LP_3, LP_4], axis=1) / norm_denom

        # Build the U part: wave[:,0] => Delta_i, plus normalized real/imag of rabi_i, rabi_r
        Delta_col = wave[:, 0:1]
        Ri_re = tf.math.real(rabi_i_cplx) / self.Rabi_i
        Ri_im = tf.math.imag(rabi_i_cplx) / self.Rabi_i
        Rr_re = tf.math.real(rabi_r_cplx) / self.Rabi_r
        Rr_im = tf.math.imag(rabi_r_cplx) / self.Rabi_r

        U_stacked = tf.concat([Delta_col, Ri_re, Ri_im, Rr_re, Rr_im], axis=1)
        U_stacked_cplx = tf.complex(U_stacked, tf.zeros_like(U_stacked)) * tf.constant(-1j, tf.complex128)

        return tf.concat([U_stacked_cplx, P_stacked], axis=1)  # [N, 9]

    def exponentials(self):
        """
        Build and exponentiate the generator at each time step:
          exponent = delta_t * ( -1j*(Gens[0]+Gens[1]) + sum_{ctrl_i} [ampl_i * Gens[2..6]] )
        Returns => [N, 2*dim^2, 2*dim^2], batch of matrix exponentials.
        """
        gens = self.VL_Generators_ansatz  # [11, 2dim^2, 2dim^2]
        aux = self.return_auxiliary_amplitudes()  # [N, 9]
        # drift = -1j * (gens[0] + gens[1])
        drift = tf.constant(-1j, dtype=tf.complex128) * (gens[0] + gens[1])

        # control gens => gens[2], gens[3], gens[4], gens[5], gens[6]
        control_gens = gens[2:7]  # shape [5, 2dim^2, 2dim^2]
        aux_u = aux[:, :5]       # [N,5], first 5 columns are for the exponent

        sum_control = tf.linalg.tensordot(aux_u, control_gens, axes=[[1],[0]])  # => [N,2dim^2,2dim^2]
        drift_expanded = drift[tf.newaxis, :, :]  # [1,2dim^2,2dim^2]
        exponent = self.delta_t * (drift_expanded + sum_control)
        # batch matrix exponential
        def single_expm(mat):
            return tf.linalg.expm(mat)
        step_exps = tf.map_fn(single_expm, exponent, fn_output_signature=tf.complex128)
        return step_exps  # [N, 2dim^2, 2dim^2]

    def propagate(self):
        """
        Multiply all step exponentials from last to first, i.e. final = E_{N-1} * ... * E_0.
        We'll implement this in pure TF using tf.scan on the reversed exponentials.
        """
        step_exps = self.exponentials()  # [N, 2dim^2, 2dim^2]
        # reverse them so step_exps_rev[0] = last
        step_exps_rev = tf.reverse(step_exps, axis=[0])

        def mul_func(acc, x):
            # multiply x * acc
            return tf.linalg.matmul(x, acc)

        # we start from identity
        eye_mat = tf.eye(2*self.dim**2, dtype=tf.complex128)
        full_stack = tf.scan(mul_func, step_exps_rev, initializer=eye_mat)
        # final result is full_stack[-1]
        return full_stack[-1]

    def metrics(self):
        """
        1) Propagate -> final operator (Van Loan prop)
        2) Compute infidelity vs self.U_target
        3) Compute adiabaticity for state |10>.
        """
        VL_propagator = self.propagate()
        U_prop = VL_propagator[0:self.dim**2, 0:self.dim**2]
        U_target = tf.cast(self.U_target, tf.complex128)

        dtrace = tf.linalg.trace(tf.linalg.matmul(U_target, U_prop, adjoint_a=True))
        dtrace_norm = tf.linalg.trace(tf.linalg.matmul(U_target, U_target, adjoint_a=True))
        infidelity = 1.0 - tf.math.real(
            dtrace * tf.math.conj(dtrace) / (dtrace_norm ** 2)
        )

        # adiabatic integral block => [0:dim^2, dim^2 : 2*dim^2]
        ad_integral = VL_propagator[0:self.dim**2, self.dim**2 : 2*self.dim**2]
        final_state_10 = tf.linalg.matmul(U_prop, self.psi_10)
        overlap = tf.linalg.matmul(
            final_state_10,
            tf.linalg.matmul(ad_integral, self.psi_10),
            adjoint_a=True
        )
        adiabaticity_metric_10 = 1.0 - (1.0 / self.duration) * overlap
        adiabaticity = tf.cast(adiabaticity_metric_10, dtype=tf.float64)

        return infidelity, adiabaticity

    def target(self):
        """
        Weighted cost => 0.2 * infidelity + 0.8 * adiabaticity
        """
        infidelity, adiabaticity = self.metrics()
        return 0.2 * infidelity + 0.8 * adiabaticity