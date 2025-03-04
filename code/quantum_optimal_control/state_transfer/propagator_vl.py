import numpy as np
import tensorflow as tf
from qutip import *


class PropagatorVL:
    def __init__(
        self, input_dim, no_of_steps, delta_t, del_total,
        Delta_1, Rabi_1, Rabi_2, Gamma_r, Gamma_i
    ):
        self.duration = no_of_steps * delta_t
        self.delta_t = delta_t
        self.Rabi_1 = Rabi_1  # Set maximum control amplitude
        self.Rabi_2 = Rabi_2  # Set maximum control amplitude
        self.Delta_1 = Delta_1  # Set maximum control amplitude
        self.del_total = del_total
        self.Gamma_r = Gamma_r
        self.Gamma_i = Gamma_i
        self.dim = 5  # Dimension of system is a 5 level atom
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps
        """
        Set initial and final states
        """
        g_prime, r_prime, g, i, r = self.nLevelAtomBasis(self.dim)
        self.psi_0 = tf.cast(g, dtype=tf.complex128)
        self.psi_t = tf.cast(r, dtype=tf.complex128)
        """
        Set up Van Loan generators
        """
        Ham_args = [g_prime, r_prime, g, i, r, self.del_total,
                    self.Delta_1, self.Rabi_1, self.Rabi_2, self.Gamma_r, self.Gamma_i]
        basis = [g_prime, r_prime, g, i, r]
        self.VL_Generators_ansatz = self.VL_Generators(
            self.Hamiltonian(Ham_args), basis, self.dim
        )

        """
        Number of control amplitudes:
        (Delta_1, real Rabi_1, real Rabi_2, imag Rabi_1, imag Rabi_2)
        """
        self.num_ctrls = 5
        """
            There are 5 control amplitudes in order:
            Delta_1, Rabi_1 amplitude, Rabi_1 phase, 
            Rabi_2 amplitude, Rabi_2 phase,
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
        self.contraction_array = self.gen_contraction_array(no_of_steps)

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
            states.append(basis(n, n_l))
        return states

    @staticmethod
    def Hamiltonian(args):
        """Ladder-system Hamiltonian: Outputs a Qutip quantum object"""
        # Set 5 level sysem operators: includes dark ground and Rydberg states
        g_prime, r_prime, g, i, r, del_total,\
            Delta_1, Rabi_1, Rabi_2, Gamma_r, Gamma_i = args

        sig_gpgp = g_prime * g_prime.dag()
        sig_gg = g * g.dag()
        sig_ii = i * i.dag()
        sig_rr = r * r.dag()
        sig_ir = i * r.dag()
        sig_gi = g * i.dag()
        sig_gpi = g_prime * i.dag()
        sig_rpr = r_prime * r.dag()
        sig_gpr = g_prime * r.dag()
        sig_gpgp = g_prime * g.dag()

        # Set projectors for finding expectation values:
        proj_g = sig_gg
        proj_i = sig_ii
        proj_r = sig_rr

        # Set Hamiltonian parts:
        H_decay = - 1j * Gamma_i / 2 * sig_ii \
            - 1j * Gamma_r / 2 * sig_rr
        H0_del_tot = - del_total * sig_rr
        H0_Delta_1 = - Delta_1 * sig_ii

        H1_re = - 1/2 * Rabi_1 * (sig_gi + sig_gi.dag())
        H1_im = - 1/2 * 1j * Rabi_1 * (sig_gi - sig_gi.dag())

        H2_re = - 1/2 * Rabi_2 * (sig_ir + sig_ir.dag())
        H2_im = - 1/2 * 1j * Rabi_2 * (sig_ir - sig_ir.dag())

        # Need to convert each Qutip Qobject into correct tf dtype
        H = [H_decay, H0_del_tot, H0_Delta_1, H1_re, H1_im, H2_re, H2_im]

        return H
    """
    Generate the Van Loan matrix from generators
    """
    @staticmethod
    def VL_Generators(Hamiltonian, basis, dim):
        # Set Hamiltonian parts:

        U_Decay = Hamiltonian[0]  # decay
        U1 = Hamiltonian[1]  # del_tot
        U2 = Hamiltonian[2]  # Delta_1
        U3_real = Hamiltonian[3]  # H1_re
        U3_imag = Hamiltonian[4]  # H1_im
        U4_real = Hamiltonian[5]  # H2_re
        U4_imag = Hamiltonian[6]  # H2_im

        Zeroes = np.zeros((dim, dim))
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
        # Projector:
        g_prime, r_prime, g, i, r = basis

        LP1 = np.block([
            [Zeroes, g * g.dag()],
            [Zeroes, Zeroes]
        ])
        LP2 = np.block([
            [Zeroes, r * r.dag()],
            [Zeroes, Zeroes]
        ])
        LP3 = np.block([
            [Zeroes, g * r.dag()],
            [Zeroes, Zeroes]
        ])
        LP4 = np.block([
            [Zeroes, r * g.dag()],
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
            Basis_matrix = self.gen_basis_matrix(
                self.no_of_steps, ctrl_amplitudes_b[:, control_num], ctrl_amplitudes_c[:, control_num]
            )
            transformed_amplitudes_temp = tf.linalg.matmul(
                Basis_matrix, self.ctrl_amplitudes_a[:, control_num:control_num + 1]
            )
            transformed_amplitudes_all.append(transformed_amplitudes_temp)
        return tf.concat((
            transformed_amplitudes_all
        ), 1
        )

    """
        regularize_amplitudes ensures that no individual amplitude exceeds 1
    """
    @staticmethod
    def regularize_amplitudes(amplitudes):
        amplitude_norms = tf.math.sqrt(
            tf.math.square(amplitudes[:, 0]) + tf.math.square(amplitudes[:, 1])
        )
        normalization_factor = tf.math.tanh(amplitude_norms) / amplitude_norms
        return tf.math.multiply(
            tf.stack([normalization_factor, normalization_factor], 1),
            amplitudes
        )
    """
        return_physical_amplitudes returns the normalized physical pulses
        (since Delta_1 does not have a real and imaginary part so need to
        do it separately) (returns float64)
    """

    def return_physical_amplitudes(self):
        # Renormalize ctl amplitudes such that -1 < Delta_1 < +1,
        # and -1 < Norm(Rabi_1) < +1, and -1 < Norm(Rabi_2) < +1 (since each
        # Rabi_1 and Rabi_2 have imgainary and real parts)
        regularized_amplitudes_Delta = tf.math.tanh(
            self.transform_amplitudes()[:, 0:1]
        )
        regularized_amplitudes_Rabi_1 = self.regularize_amplitudes(
            self.transform_amplitudes()[:, 1:3]
        )
        regularized_amplitudes_Rabi_2 = self.regularize_amplitudes(
            self.transform_amplitudes()[:, 3:5]
        )
        regularized_amplitudes = tf.concat((
            regularized_amplitudes_Delta,
            regularized_amplitudes_Rabi_1,
            regularized_amplitudes_Rabi_2
        ), 1
        )
        return regularized_amplitudes
    """
    return_auxiliary_amplitudes outputs all variables needed for the VL 
    generator: U diagonal, P off diagonal block matrices
    P park of VL generators:|D><D|= Rabi_2^2|g><g| + Rabi_1^2|r><r|
    - Rabi_2 X Rabi_1 |g><r| - Rabi_2* X Rabi_1* |r><g|
    """

    def return_auxiliary_amplitudes(self):
        amplitudes_U = self.return_physical_amplitudes()

        Rabi_1_complex = tf.math.scalar_mul(
            self.Rabi_1,
            tf.dtypes.complex(
                amplitudes_U[:, 1:2], amplitudes_U[:, 2:3]
            )
        )
        Rabi_2_complex = tf.math.scalar_mul(
            self.Rabi_2,
            tf.dtypes.complex(
                amplitudes_U[:, 3:4], amplitudes_U[:, 4:5]
            )
        )

        # |Rabi_2|^2:
        LP_1 = tf.math.multiply(
            Rabi_2_complex,
            tf.math.conj(Rabi_2_complex)
        )
        # |Rabi_1|^2:
        LP_2 = tf.math.multiply(
            Rabi_1_complex,
            tf.math.conj(Rabi_1_complex)
        )
        # - Rabi_1 X Rabi_2:
        LP_3 = - tf.math.multiply(
            Rabi_1_complex,
            Rabi_2_complex
        )
        # - Rabi_1* X Rabi_2*:
        LP_4 = - tf.math.multiply(
            tf.math.conj(Rabi_1_complex),
            tf.math.conj(Rabi_2_complex)
        )
        norm = tf.math.add(LP_1, LP_2)

        amplitudes_P = tf.math.divide(
            tf.concat(
                [LP_1, LP_2, LP_3, LP_4], 1
            ), norm
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
            VL_Gens[0] + VL_Gens[1] + tf.linalg.tensordot(
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
        psi_0 = self.psi_0
        psi_t = self.psi_t
        """
            infidelity part in the target
        """
        U_prop = VL_propagator[0:self.dim, 0:self.dim]
        final_state = tf.linalg.matmul(
            U_prop, psi_0
        )
        overlap = tf.linalg.matmul(
            psi_t, final_state, adjoint_a=True
        )
        infidelity = 1 - tf.math.real(tf.math.conj(overlap) * overlap)

        """
            Adiabatic metric
        """
        ad_integral = VL_propagator[0:self.dim, self.dim:2 * self.dim]

        adiabaticity_metric = 1 - (1 / self.duration) * tf.linalg.matmul(
            final_state,
            tf.linalg.matmul(
                ad_integral, psi_0
            ),
            adjoint_a=True
        )

        adiabaticity = tf.cast(adiabaticity_metric, dtype=tf.float64)

        return infidelity, adiabaticity
    """
        target computes the final cost function
    """
    @tf.function
    def target(self):
        metrics = self.metrics()
        return 0.6 * metrics[0] + 0.4 * metrics[1]
