import numpy as np
import scipy.signal
import tensorflow as tf
from qutip import *


class PropagatorVL:
    def __init__(
        self, input_dim, no_of_steps, pad, f_std, delta_t, del_total, V_int,
        Delta_i, Rabi_i, Rabi_r, Gammas_all
    ):
        self.duration = (no_of_steps + 2 * pad) * delta_t
        self.delta_t = delta_t
        self.Vint = V_int
        self.Rabi_i = Rabi_i  # Set maximum control amplitude
        self.Rabi_r = Rabi_r  # Set maximum control amplitude
        self.Delta_i = Delta_i  # Set maximum control amplitude
        self.del_total = del_total
        self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd = Gammas_all
        self.dim = 4  # Dimension of each atom
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps
        """ generate a Gaussian filter array (in frequency domain) """
        self.padding = pad  # sets extra number of points on either side of pulse
        self.freqs = np.fft.fftfreq(no_of_steps, d=delta_t)  # set frequency spectrum
        self.std = f_std/(self.freqs[1] - self.freqs[0])  # Set standard deviation of Gaussian filter
        self.Gaussian_filter = tf.constant(
            scipy.signal.gaussian(no_of_steps + 2 * self.padding, self.std),
            dtype=tf.complex128
        )
        """
        Set initial and final states and target gate
        """
        g_0, g_1, i, r = self.nLevelAtomBasis(self.dim)  # basis per atom
        self.psi_pp = tf.cast(1/2 * (
            tensor(g_0, g_0) + tensor(g_0, g_1) +
            tensor(g_1, g_0) + tensor(g_1, g_1)
        ), dtype=tf.complex128)

        self.psi_10 = tf.cast(tensor(g_1, g_0), dtype=tf.complex128)
        self.psi_11 = tf.cast(tensor(g_1, g_1), dtype=tf.complex128)
        self.psi_01 = tf.cast(tensor(g_0, g_1), dtype=tf.complex128)
        self.U_target = tensor(g_0, g_0) * tensor(g_0, g_0).dag() \
            + tensor(g_0, g_1) * tensor(g_0, g_1).dag() \
            + tensor(g_1, g_0) * tensor(g_1, g_0).dag() \
            - tensor(g_1, g_1) * tensor(g_1, g_1).dag()
        """
        Set up Van Loan generators
        """
        Ham_args = [g_0, g_1, i, r, self.dim, self.del_total, self.Vint,
                    self.Delta_i, self.Rabi_i, self.Rabi_r, self.Gamma_10,
                    self.Gamma_i1, self.Gamma_ri, self.Gamma_rd]
        basis = [g_0, g_1, i, r]
        self.VL_Generators_ansatz = self.VL_Generators(
            self.Hamiltonian(Ham_args), basis, self.dim
        )

        """
        Number of control amplitudes: global for both atoms
        (Delta_i, mag Rabi_i, phase Rabi_r, mag Rabi_i, phase Rabi_r)
        """
        self.num_ctrls = 5
        """
            There are 5 control amplitudes in order:
            Delta_i, Rabi_i amplitude, Rabi_i phase, 
            Rabi_r amplitude, Rabi_r phase,
        """
        # Gaussian amplitudes:
        self.ctrl_amplitudes_a = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64
                     ), dtype=tf.float64
        )
        # Gaussian centers:
        self.ctrl_amplitudes_b = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64
                     ), dtype=tf.float64
        )
        # Gaussian STD:
        self.ctrl_amplitudes_c = tf.Variable(
            tf.zeros([input_dim, self.num_ctrls], dtype=tf.float64
                     ), dtype=tf.float64
        )
        # All control amplitudes together
        self.ctrl_amplitudes_all = tf.concat(
            [self.ctrl_amplitudes_a, self.ctrl_amplitudes_b,
             self.ctrl_amplitudes_c], 1
        )

        """
            self.contraction_array determines the neccessity for the extra
            matrix multiplication step in the recursive method self.propagate()
            when the intermediate computation array has length not divisible
            by 2
        """
        self.contraction_array = self.gen_contraction_array(
            no_of_steps + 2 * self.padding
        )

    @staticmethod
    def gen_contraction_array(no_of_intervals):
        if no_of_intervals > 1:
            return (
                [bool(np.mod(no_of_intervals, 2))] +
                PropagatorVL.gen_contraction_array(
                    np.floor(no_of_intervals / 2)
                )
            )
        return []

    """
        nLevelAtomBasis creates a basis set for an n level atom as qutip
        quantum objects
    """
    @staticmethod
    def nLevelAtomBasis(n):
        states = []
        for n_l in range(0, n):
            states.append(qutip.basis(n, n_l))
        return states

    @staticmethod
    def Hamiltonian(args):
        """Ladder-system Hamiltonian: Outputs a Qutip quantum object"""
        # Set 5 level sysem operators: includes dark ground and Rydberg states
        g_0, g_1, i, r, dim, del_total, V_int, \
            Delta_i, Rabi_i, Rabi_r, Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd = args

        sig_00 = g_0 * g_0.dag()
        sig_11 = g_1 * g_1.dag()
        sig_ii = i * i.dag()
        sig_rr = r * r.dag()
        sig_01 = g_0 * g_1.dag()
        sig_1i = g_1 * i.dag()
        sig_ir = i * r.dag()

        # Set decay hamiltonian: (effetive non hermitian part)
        H_SQ_decay = - 1j * Gamma_i1 / 2 * sig_ii \
            - 1j * Gamma_10 / 2 * sig_11 \
            - 1j * (Gamma_ri + Gamma_rd) / 2 * sig_rr

        # Set single qubit hamiltonian parts:
        H_SQ_del = - del_total * sig_rr
        H_SQ_Delta_i = - Delta_i * sig_ii
        H_SQ_Rabi_i_re = - 1/2 * Rabi_i * (sig_1i + sig_1i.dag())
        H_SQ_Rabi_i_im = - 1j/2 * Rabi_i * (sig_1i - sig_1i.dag())
        H_SQ_Rabi_r_re = - 1/2 * Rabi_r * (sig_ir + sig_ir.dag())
        H_SQ_Rabi_r_im = - 1j/2 * Rabi_r * (sig_ir - sig_ir.dag())

        # Set two qubit static terms
        H_TQ_0 = tensor(identity(dim), H_SQ_decay) + \
            tensor(H_SQ_decay, identity(dim)) + \
            V_int * tensor(sig_rr, sig_rr)

        # Set two qubit detunings:
        H_TQ_Delta = tensor(identity(dim), H_SQ_Delta_i) + tensor(H_SQ_Delta_i, identity(dim))
        H_TQ_del_total = tensor(identity(dim), H_SQ_del) + tensor(H_SQ_del, identity(dim))

        # Real part
        # Set drive 1:(use same power on each qubit))
        H_TQ_Rabi_i_re = tensor(H_SQ_Rabi_i_re, identity(dim)) + \
            tensor(identity(dim), H_SQ_Rabi_i_re)
        # Set second qubit drive 1:
        H_TQ_Rabi_r_re = tensor(H_SQ_Rabi_r_re, identity(dim)) + \
            tensor(identity(dim), H_SQ_Rabi_r_re)

        # Imaginary part
        H_TQ_Rabi_i_im = tensor(H_SQ_Rabi_i_im, identity(dim)) + \
            tensor(identity(dim), H_SQ_Rabi_i_im)
        # Set second qubit drive 1:
        H_TQ_Rabi_r_im = tensor(H_SQ_Rabi_r_im, identity(dim)) + \
            tensor(identity(dim), H_SQ_Rabi_r_im)

        # Need to convert each Qutip Qobject into correct tf dtype
        H = [H_TQ_0, H_TQ_del_total, H_TQ_Delta,
             H_TQ_Rabi_i_re, H_TQ_Rabi_i_im,
             H_TQ_Rabi_r_re, H_TQ_Rabi_r_im]

        return H

    """
    Generate the Van Loan matrix from generators
    """
    @staticmethod
    def VL_Generators(Hamiltonian, basis, dim):
        # Set Hamiltonian parts:

        U_Decay = Hamiltonian[0]  # decay (and V_int)
        U1 = Hamiltonian[1]  # del_tot
        U2 = Hamiltonian[2]  # Delta_i
        U3_real = Hamiltonian[3]  # H1_re
        U3_imag = Hamiltonian[4]  # H1_im
        U4_real = Hamiltonian[5]  # H2_re
        U4_imag = Hamiltonian[6]  # H2_im

        # Two atoms have total Hamiltonian dimension dim**2:
        Zeroes = np.zeros((dim**2, dim**2))

        # Hamiltonian:
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
        # Projector for dark state: (for atom #1)
        g_0, g_1, i, r = basis

        LP1 = np.block([
            [Zeroes, tensor(g_1 * g_1.dag(), g_0 * g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP2 = np.block([
            [Zeroes, tensor(r * r.dag(), g_0 * g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP3 = np.block([
            [Zeroes, tensor(g_1 * r.dag(), g_0 * g_0.dag())],
            [Zeroes, Zeroes]
        ])
        LP4 = np.block([
            [Zeroes, tensor(r * g_1.dag(), g_0 * g_0.dag())],
            [Zeroes, Zeroes]
        ])
        return tf.stack(
            [LU_decay, LU1, LU2, LU3_real, LU3_imag, LU4_real, LU4_imag,
             LP1, LP2, LP3, LP4]
        )

    """
    Generates the time trace of the different Gaussian modes 
    (the length of b and c are the number of Gaussian modes)
    """
    @staticmethod
    def gen_basis_matrix(no_of_steps, amp_b, amp_c):
        # Set times from 0 to 1 (to scale later)
        times = np.linspace(-1, 1, no_of_steps)

        def Gaussian(x):
            b, c = x[0], x[1]
            return tf.math.exp(
                - tf.math.divide(
                    tf.math.square(times - b), tf.math.square(c)
                )
            )
        Gaussian_modes = tf.map_fn(Gaussian, (amp_b, amp_c), dtype=tf.float64)
        return tf.transpose(tf.stack(Gaussian_modes))

    """
        transform_amplitudes performs a transformation to a basis
        (e.g. Legendre, Gaussian, Fourier etc.)
    """

    def transform_amplitudes(self):
        # Ensure that ctrl_amplitudes_b and ctrl_amplitudes_c are always
        # between -1 and 1 (so the Gaussians peak with the plotted time and
        # the width is not too large):
        ctrl_amplitudes_b = tf.math.tanh(self.ctrl_amplitudes_b)
        ctrl_amplitudes_c = tf.math.tanh(self.ctrl_amplitudes_c)

        transformed_amplitudes_all = []
        # Loop for creating each matrix of time traced Gaussian modes for
        # each control amplitude
        for control_num in range(self.num_ctrls):
            # Create Gaussian modes with variable center and width
            Basis_matrix = self.gen_basis_matrix(
                self.no_of_steps, ctrl_amplitudes_b[:, control_num],
                ctrl_amplitudes_c[:, control_num]
            )
            # Multiply each Gaussian with variable amplitude
            transformed_amplitudes_temp = tf.linalg.matmul(
                Basis_matrix,
                self.ctrl_amplitudes_a[:, control_num:control_num + 1]
            )
            transformed_amplitudes_all.append(transformed_amplitudes_temp)
        return tf.concat((
            transformed_amplitudes_all
        ), 1
        )

    """
        regularize_amplitudes ensures that no individual amplitude exceeds 1
    """

    def regularize_amplitudes(self):
        # Renormalize ctl amplitudes such that -1 < Delta_i < +1,
        # and -1 < Norm(Rabi_i) < +1, and -1 < Norm(Rabi_r) < +1 (since each
        # Rabi_i and Rabi_r have imgainary and real parts)
        times = np.linspace(0, self.duration, self.no_of_steps)

        # Constant detuning:
        regularized_amplitudes_Delta = 0 * self.transform_amplitudes()[:, 0:1] - 1

        # regularized_amplitudes_Delta = tf.math.tanh(
        #     self.transform_amplitudes()[:,0:1]
        #     )

        regularized_amplitudes_Rabi_i_mag = 1 - tf.math.exp(
            - tf.math.square(self.transform_amplitudes()[:, 1:2])
        )

        regularized_amplitudes_Rabi_i_phase = tf.math.tanh(
            self.transform_amplitudes()[:, 2:3]
        )

        regularized_amplitudes_Rabi_r_mag = 1 - tf.math.exp(
            - tf.math.square(self.transform_amplitudes()[:, 3:4])
        )

        regularized_amplitudes_Rabi_r_phase = tf.math.tanh(
            self.transform_amplitudes()[:, 4:5]
        )

        regularized_amplitudes = tf.concat((
            regularized_amplitudes_Delta,
            regularized_amplitudes_Rabi_i_mag, regularized_amplitudes_Rabi_i_phase,
            regularized_amplitudes_Rabi_r_mag, regularized_amplitudes_Rabi_r_phase
        ), 1
        )
        return regularized_amplitudes

    """ filter_amplitudes filters the frequency components of the waveform """
    @staticmethod
    def filter_amplitudes(amplitudes, filter_function_array, padding):
        padded_amplitudes = tf.pad(tf.cast(amplitudes, dtype=tf.complex128),
                                   tf.constant([[padding, padding], [0, 0]]), "CONSTANT"
                                   )

        # Delta:
        frequency_amplitudes_Delta = tf.signal.fftshift(tf.signal.fft(
            padded_amplitudes[:, 0]
        ))
        filtered_amplitudes_Delta = tf.signal.ifft(tf.signal.ifftshift(
            tf.math.multiply(filter_function_array, frequency_amplitudes_Delta)
        ))
        # Rabi_i
        frequency_amplitudes_Rabi_i_mag = tf.signal.fftshift(tf.signal.fft(
            padded_amplitudes[:, 1]
        ))
        filtered_amplitudes_Rabi_i_mag = tf.signal.ifft(tf.signal.ifftshift(
            tf.math.multiply(filter_function_array, frequency_amplitudes_Rabi_i_mag)
        ))
        frequency_amplitudes_Rabi_i_phase = tf.signal.fftshift(tf.signal.fft(
            padded_amplitudes[:, 2]
        ))
        filtered_amplitudes_Rabi_i_phase = tf.signal.ifft(tf.signal.ifftshift(
            tf.math.multiply(filter_function_array, frequency_amplitudes_Rabi_i_phase)
        ))
        # Rabi_r
        frequency_amplitudes_Rabi_r_mag = tf.signal.fftshift(tf.signal.fft(
            padded_amplitudes[:, 3]
        ))
        filtered_amplitudes_Rabi_r_mag = tf.signal.ifft(tf.signal.ifftshift(
            tf.math.multiply(filter_function_array, frequency_amplitudes_Rabi_r_mag)
        ))
        frequency_amplitudes_Rabi_r_phase = tf.signal.fftshift(tf.signal.fft(
            padded_amplitudes[:, 4]
        ))
        filtered_amplitudes_Rabi_r_phase = tf.signal.ifft(tf.signal.ifftshift(
            tf.math.multiply(filter_function_array, frequency_amplitudes_Rabi_r_phase)
        ))

        # cast all values into complex128
        filtered_amplitudes_Delta = tf.cast(filtered_amplitudes_Delta,
                                            dtype=tf.float64)
        filtered_amplitudes_Rabi_i_mag = tf.cast(filtered_amplitudes_Rabi_i_mag,
                                                 dtype=tf.float64)
        filtered_amplitudes_Rabi_i_phase = tf.cast(filtered_amplitudes_Rabi_i_phase,
                                                   dtype=tf.float64)
        filtered_amplitudes_Rabi_r_mag = tf.cast(filtered_amplitudes_Rabi_r_mag,
                                                 dtype=tf.float64)
        filtered_amplitudes_Rabi_r_phase = tf.cast(filtered_amplitudes_Rabi_r_phase,
                                                   dtype=tf.float64)
        return tf.stack([filtered_amplitudes_Delta,
                         filtered_amplitudes_Rabi_i_mag,
                         filtered_amplitudes_Rabi_i_phase,
                         filtered_amplitudes_Rabi_r_mag,
                         filtered_amplitudes_Rabi_r_phase
                         ], axis=1)

    """
        return_physical_amplitudes transforms the input array
        of normalized amplitudes into physical control amplitudes
        that are filtered (returns float64)
    """

    def return_physical_amplitudes(self):
        transformed_norm_amplitudes = self.regularize_amplitudes()
        filtered_amplitudes = self.filter_amplitudes(
            transformed_norm_amplitudes, self.Gaussian_filter, self.padding)
        return filtered_amplitudes

    """
    return_auxiliary_amplitudes outputs all variables needed for the VL 
    generator: U diagonal, P off diagonal block matrices
    P park of VL generators:|D><D|= Rabi_r^2|g><g| + Rabi_i^2|r><r|
    - Rabi_r X Rabi_i |g><r| - Rabi_r* X Rabi_i* |r><g|
    """

    def return_auxiliary_amplitudes(self):
        amplitudes_phasemag = self.return_physical_amplitudes()
        # combines amplitude and phase to make complex values for Rabi freqs.
        Rabi_i_mag = tf.math.scalar_mul(
            self.Rabi_i, amplitudes_phasemag[:, 1:2]
        )
        Rabi_i_phase = tf.math.scalar_mul(
            np.pi, amplitudes_phasemag[:, 2:3]
        )
        Rabi_i_complex = tf.dtypes.complex(
            Rabi_i_mag * tf.math.cos(Rabi_i_phase), Rabi_i_mag * tf.math.sin(Rabi_i_phase)
        )

        Rabi_r_mag = tf.math.scalar_mul(
            self.Rabi_r, amplitudes_phasemag[:, 3:4]
        )
        Rabi_r_phase = tf.math.scalar_mul(
            np.pi, amplitudes_phasemag[:, 4:5]
        )
        Rabi_r_complex = tf.dtypes.complex(
            Rabi_r_mag * tf.math.cos(Rabi_r_phase), Rabi_r_mag * tf.math.sin(Rabi_r_phase)
        )

        # |Rabi_r|^2:
        LP_1 = tf.math.multiply(
            Rabi_r_complex,
            tf.math.conj(Rabi_r_complex)
        )
        # |Rabi_i|^2:
        LP_2 = tf.math.multiply(
            Rabi_i_complex,
            tf.math.conj(Rabi_i_complex)
        )
        # - Rabi_i X Rabi_r:
        LP_3 = - tf.math.multiply(
            Rabi_i_complex,
            Rabi_r_complex
        )
        # - Rabi_i* X Rabi_r*:
        LP_4 = - tf.math.multiply(
            tf.math.conj(Rabi_i_complex),
            tf.math.conj(Rabi_r_complex)
        )
        norm = tf.math.add(LP_1, LP_2)

        amplitudes_P = tf.math.divide(
            tf.concat(
                [LP_1, LP_2, LP_3, LP_4], 1
            ), norm
        )

        amplitudes_U = tf.concat(
            [amplitudes_phasemag[:, 0:1],
             tf.math.real(Rabi_i_complex)/self.Rabi_i,
             tf.math.imag(Rabi_i_complex)/self.Rabi_i,
             tf.math.real(Rabi_r_complex)/self.Rabi_r,
             tf.math.imag(Rabi_r_complex)/self.Rabi_r
             ], 1
        )
        return tf.concat([
            - 1j * tf.cast(amplitudes_U, dtype=tf.complex128),
            tf.cast(amplitudes_P, dtype=tf.complex128)
        ], 1
        )

    """
        exponentials() computes a vector matrix exponential after multiplying
        each self.ctrl_amplitudes row with a the vector of matrices in
        self.generators
    """

    def exponentials(self):
        VL_Gens = self.VL_Generators_ansatz
        # Set all regularized ctrl amplitdues into single 2D array
        auxiliary_amplitudes = self.return_auxiliary_amplitudes()
        # First two elements in the Hamiltonian list are not multiplied by a
        # control amplitude (Drift part): decay terms plus del_tot term
        exponents = self.delta_t * (
            -1j * VL_Gens[0] - 1j * VL_Gens[1] + tf.linalg.tensordot(
                auxiliary_amplitudes, VL_Gens[2:], 1
            )
        )
        return tf.linalg.expm(exponents)

    """
        propagate  computes the final propagator by recursively multiplying
        each odd element in the list of matrices with each even element --
        if the length of the array is not divisible by 2 an extra computation
        step is added
    """

    def propagate(self):
        step_exps = self.exponentials()
        for is_odd in self.contraction_array:
            if is_odd:
                odd_exp = step_exps[-1, :, :]
                step_exps = tf.linalg.matmul(
                    step_exps[1::2, :, :], step_exps[0:-1:2, :, :]
                )
                step_exps = tf.concat([
                    step_exps[0:-1, :, :],
                    [tf.linalg.matmul(odd_exp, step_exps[-1, :, :])]
                ], 0)
            else:
                step_exps = tf.linalg.matmul(
                    step_exps[1::2, :, :], step_exps[0::2, :, :]
                )
        return tf.squeeze(step_exps)

    """
        metrics computes the final state and adiabaticity metrics
    """

    def metrics(self):
        VL_propagator = self.propagate()
        """
            infidelity part in the target: unitary gate
        """
        U_prop = VL_propagator[0:self.dim**2, 0:self.dim**2]
        U_target = tf.cast(self.U_target, dtype=tf.complex128)

        dtrace = tf.linalg.trace(
            tf.linalg.matmul(U_target, U_prop, adjoint_a=True)
        )
        dtrace_norm = tf.linalg.trace(
            tf.linalg.matmul(U_target, U_target, adjoint_a=True)
        )

        infidelity = 1 - tf.math.real(
            dtrace * tf.math.conj(dtrace) /
            (dtrace_norm ** 2)
        )

        """
            Adiabatic metric
        """
        ad_integral = VL_propagator[0:self.dim**2, self.dim**2:2 * self.dim**2]

        # Single atom dark state: state |10> stays in first atom dark state |D0>
        psi_initial_10 = self.psi_10
        # psi_initial_11 = self.psi_11

        final_state_10 = tf.linalg.matmul(
            U_prop, psi_initial_10
        )
        # final_state_11 = tf.linalg.matmul(
        #     U_prop, psi_initial_11
        # )
        
        # make |1> initial state follow adiabatic dark state to r, then back to 1:
        adiabaticity_metric_10 = 1 - (1 / self.duration) * tf.linalg.matmul(
            final_state_10,
            tf.linalg.matmul(
                ad_integral, psi_initial_10
            ),
            adjoint_a=True
        )

        # #make |11> initial state follow adiabatic dark state
        # adiabaticity_metric_11 = 1  - (1 / self.duration) * tf.linalg.matmul(
        #     final_state_11,
        #     tf.linalg.matmul(
        #         ad_integral, psi_initial_11
        #     ),
        #     adjoint_a = True
        # )

        # adiabaticity = tf.cast(
        #     (0.5 * adiabaticity_metric_10 + 0.5 * adiabaticity_metric_11),
        #     dtype = tf.float64)

        # For state to state transfer:
        adiabaticity = tf.cast(
            adiabaticity_metric_10,
            dtype=tf.float64)
        return infidelity, adiabaticity
    """
        target computes the final cost function
    """

    def target(self):
        metrics = self.metrics()
        return 0.2 * metrics[0] + 0.8 * metrics[1]
    