import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import vmap, jit
from functools import partial

# ----------------------------------------------------------------------
# Helper functions (analogous to those in two_qubit propagator)
# ----------------------------------------------------------------------
def dag(mat: jnp.ndarray) -> jnp.ndarray:
    """Conjugate transpose (dagger)."""
    return jnp.conj(mat.T)

def basis(dim: int, idx: int) -> jnp.ndarray:
    """Generate a unit vector of length 'dim' with 1 at position 'idx'."""
    e = jnp.zeros((dim,), dtype=jnp.complex128)
    return e.at[idx].set(1.0)

def outer(ket: jnp.ndarray) -> jnp.ndarray:
    """Compute the outer product |ket><ket|."""
    return jnp.outer(ket, jnp.conj(ket))

def gaussian_array(M: int, std: float) -> jnp.ndarray:
    """Generate a Gaussian array of length M with given standard deviation."""
    x = jnp.arange(M, dtype=jnp.float64)
    c = 0.5 * (M - 1)
    return jnp.exp(-0.5 * ((x - c) / std)**2)

# ----------------------------------------------------------------------
# State Transfer Van Loan Propagator in JAX (5-level atom)
# ----------------------------------------------------------------------
class PropagatorVL:
    def __init__(
        self,
        input_dim: int,
        no_of_steps: int,
        delta_t: float,
        del_total: float,
        Delta_1: float,
        Rabi_1: float,
        Rabi_2: float,
        Gamma_r: float,
        Gamma_i: float
    ):
        # Set simulation parameters
        self.duration = no_of_steps * delta_t
        self.delta_t = delta_t
        self.Rabi_1 = Rabi_1
        self.Rabi_2 = Rabi_2
        self.Delta_1 = Delta_1
        self.del_total = del_total
        self.Gamma_r = Gamma_r
        self.Gamma_i = Gamma_i
        self.dim = 5  # 5-level atom
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps

        # Set initial and target states using the 5-level basis
        g_prime, r_prime, g, i_state, r = self.nLevelAtomBasis(self.dim)
        self.psi_0 = g        # initial state is |g>
        self.psi_t = r        # target state is |r>

        # Build Van Loan generators from the Hamiltonian parts
        Ham_args = [g_prime, r_prime, g, i_state, r,
                    self.del_total, self.Delta_1,
                    self.Rabi_1, self.Rabi_2,
                    self.Gamma_r, self.Gamma_i]
        basis_list = [g_prime, r_prime, g, i_state, r]
        self.VL_Generators_ansatz = self.VL_Generators(self.Hamiltonian(Ham_args), basis_list, self.dim)

        # Number of control amplitudes (Delta_1, Rabi_1 re/imag, Rabi_2 re/imag)
        self.num_ctrls = 5

        # Generate contraction array for odd/even recursion in propagate()
        self.contraction_array = self.gen_contraction_array(no_of_steps)

    @staticmethod
    def gen_contraction_array(no_of_intervals: int):
        """
        Identical to the TF method:
        If 'no_of_intervals' > 1, store if it's odd or even, then
        recurse on floor(no_of_intervals/2).
        """
        if no_of_intervals > 1:
            return [bool(no_of_intervals % 2)] + PropagatorVL.gen_contraction_array(no_of_intervals // 2)
        else:
            return []

    @staticmethod
    def nLevelAtomBasis(n: int):
        """Create a basis set for an n-level atom."""
        states = []
        for idx in range(n):
            states.append(basis(n, idx))
        return states

    @staticmethod
    def Hamiltonian(args):
        """
        Constructs the ladder-system Hamiltonian for a 5-level atom.
        Args order: [g_prime, r_prime, g, i_state, r, del_total, Delta_1, Rabi_1, Rabi_2, Gamma_r, Gamma_i]
        """
        g_prime, r_prime, g, i_state, r, del_total, Delta_1, Rabi_1, Rabi_2, Gamma_r, Gamma_i = args

        # Compute operators (note: the last assignment overwrites the first sig_gpgp as in the original code)
        sig_gpgp = outer(g_prime)
        sig_gg = outer(g)
        sig_ii = outer(i_state)
        sig_rr = outer(r)
        sig_ir = jnp.outer(i_state, jnp.conj(r))
        sig_gi = jnp.outer(g, jnp.conj(i_state))
        sig_gpi = jnp.outer(g_prime, jnp.conj(i_state))
        sig_rpr = jnp.outer(r_prime, jnp.conj(r))
        sig_gpr = jnp.outer(g_prime, jnp.conj(r))
        sig_gpgp = jnp.outer(g_prime, jnp.conj(g))  # Overwrite as in original

        # Define projectors (used for expectation values)
        proj_g = sig_gg
        proj_i = sig_ii
        proj_r = sig_rr

        # Hamiltonian parts
        H_decay = -1j * Gamma_i / 2.0 * sig_ii - 1j * Gamma_r / 2.0 * sig_rr
        H0_del_tot = - del_total * sig_rr
        H0_Delta_1 = - Delta_1 * sig_ii
        H1_re = -0.5 * Rabi_1 * (sig_gi + dag(sig_gi))
        H1_im = -0.5 * 1j * Rabi_1 * (sig_gi - dag(sig_gi))
        H2_re = -0.5 * Rabi_2 * (sig_ir + dag(sig_ir))
        H2_im = -0.5 * 1j * Rabi_2 * (sig_ir - dag(sig_ir))
        return [H_decay, H0_del_tot, H0_Delta_1, H1_re, H1_im, H2_re, H2_im]

    @staticmethod
    def VL_Generators(Hamiltonian, basis, dim: int):
        """
        Generate the Van Loan generators as a stack of block matrices.
        Constructs 11 generators: 7 from the Hamiltonian parts and 4 projector parts.
        """
        U_Decay = Hamiltonian[0]
        U1 = Hamiltonian[1]
        U2 = Hamiltonian[2]
        U3_real = Hamiltonian[3]
        U3_imag = Hamiltonian[4]
        U4_real = Hamiltonian[5]
        U4_imag = Hamiltonian[6]
        Zeroes = jnp.zeros((dim, dim), dtype=jnp.complex128)
        LU_decay = jnp.block([[U_Decay, Zeroes],
                              [Zeroes, U_Decay]])
        LU1 = jnp.block([[U1, Zeroes],
                         [Zeroes, U1]])
        LU2 = jnp.block([[U2, Zeroes],
                         [Zeroes, U2]])
        LU3_real = jnp.block([[U3_real, Zeroes],
                              [Zeroes, U3_real]])
        LU3_imag = jnp.block([[U3_imag, Zeroes],
                              [Zeroes, U3_imag]])
        LU4_real = jnp.block([[U4_real, Zeroes],
                              [Zeroes, U4_real]])
        LU4_imag = jnp.block([[U4_imag, Zeroes],
                              [Zeroes, U4_imag]])
        g_prime, r_prime, g, i_state, r = basis
        LP1 = jnp.block([[Zeroes, outer(g)],
                         [Zeroes, Zeroes]])
        LP2 = jnp.block([[Zeroes, outer(r)],
                         [Zeroes, Zeroes]])
        LP3 = jnp.block([[Zeroes, jnp.outer(g, jnp.conj(r))],
                         [Zeroes, Zeroes]])
        LP4 = jnp.block([[Zeroes, jnp.outer(r, jnp.conj(g))],
                         [Zeroes, Zeroes]])
        return jnp.stack([LU_decay, LU1, LU2, LU3_real, LU3_imag, LU4_real, LU4_imag,
                          LP1, LP2, LP3, LP4], axis=0)

    @staticmethod
    def gen_basis_matrix(no_of_steps: int, amp_b: jnp.ndarray, amp_c: jnp.ndarray):
        """
        Generate the time trace of a Gaussian mode:
        exp(-((t - b)^2 / c^2)) with t in [-1, 1].
        Returns a matrix of shape [no_of_steps, input_dim].
        """
        times = jnp.linspace(-1.0, 1.0, no_of_steps)
        def Gaussian(params):
            b, c = params
            return jnp.exp(-((times - b)**2) / (c**2))
        bc = jnp.stack([amp_b, amp_c], axis=1)
        Gaussian_modes = vmap(Gaussian)(bc)
        return Gaussian_modes.T

    def transform_amplitudes(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Transform control amplitudes to time-domain waveforms using Gaussian basis modes.
        Each control channel (of total self.num_ctrls) is transformed separately.
        """
        ctrl_amplitudes_b = jnp.tanh(ctrl_b)
        ctrl_amplitudes_c = jnp.tanh(ctrl_c)
        transformed_amplitudes_all = []
        for control_num in range(self.num_ctrls):
            Basis_matrix = self.gen_basis_matrix(self.no_of_steps,
                                                 ctrl_amplitudes_b[:, control_num],
                                                 ctrl_amplitudes_c[:, control_num])
            transformed_amplitudes_temp = jnp.matmul(Basis_matrix,
                                                     ctrl_a[:, control_num:control_num+1])
            transformed_amplitudes_all.append(transformed_amplitudes_temp)
        return jnp.concatenate(transformed_amplitudes_all, axis=1)

    @staticmethod
    def regularize_amplitudes(amplitudes: jnp.ndarray):
        """
        Regularize amplitudes so that the norm of the two components (for a complex number)
        does not exceed 1.
        """
        amplitude_norms = jnp.sqrt(amplitudes[:, 0]**2 + amplitudes[:, 1]**2)
        # Avoid division by zero
        normalization_factor = jnp.where(amplitude_norms == 0, 1.0, jnp.tanh(amplitude_norms) / amplitude_norms)
        return normalization_factor[:, None] * amplitudes

    def return_physical_amplitudes(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Return the normalized physical control pulses.
        For Delta_1: apply tanh directly.
        For Rabi_1 and Rabi_2: regularize the complex (re, im) components.
        """
        transformed = self.transform_amplitudes(ctrl_a, ctrl_b, ctrl_c)
        regularized_amplitudes_Delta = jnp.tanh(transformed[:, 0:1])
        regularized_amplitudes_Rabi_1 = self.regularize_amplitudes(transformed[:, 1:3])
        regularized_amplitudes_Rabi_2 = self.regularize_amplitudes(transformed[:, 3:5])
        return jnp.concatenate([regularized_amplitudes_Delta,
                                regularized_amplitudes_Rabi_1,
                                regularized_amplitudes_Rabi_2], axis=1)

    def return_auxiliary_amplitudes(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Generate the auxiliary amplitudes used in the Van Loan generators.
        Converts the physical amplitudes into complex control parameters and
        computes the projector part.
        """
        amplitudes_U = self.return_physical_amplitudes(ctrl_a, ctrl_b, ctrl_c)
        Rabi_1_complex = self.Rabi_1 * (amplitudes_U[:, 1:2] + 1j * amplitudes_U[:, 2:3])
        Rabi_2_complex = self.Rabi_2 * (amplitudes_U[:, 3:4] + 1j * amplitudes_U[:, 4:5])
        LP_1 = Rabi_2_complex * jnp.conj(Rabi_2_complex)
        LP_2 = Rabi_1_complex * jnp.conj(Rabi_1_complex)
        LP_3 = - Rabi_1_complex * Rabi_2_complex
        LP_4 = - jnp.conj(Rabi_1_complex) * jnp.conj(Rabi_2_complex)
        norm = LP_1 + LP_2
        amplitudes_P = jnp.concatenate([LP_1, LP_2, LP_3, LP_4], axis=1) / norm
        return jnp.concatenate([-1j * amplitudes_U.astype(jnp.complex128),
                                amplitudes_P.astype(jnp.complex128)], axis=1)

    def exponentials(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Compute the matrix exponentials for each time step.
        Combines drift (generators 0 and 1) with time-dependent controls (generators 2 onward).
        """
        VL_Gens = self.VL_Generators_ansatz  # shape: [11, 2*dim, 2*dim]
        aux = self.return_auxiliary_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # shape: [no_of_steps, 9]
        exponent = self.delta_t * (VL_Gens[0] + VL_Gens[1] +
                                   jnp.tensordot(aux, VL_Gens[2:], axes=[[1],[0]]))
        return vmap(jla.expm)(exponent)

    def propagate(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Follows the same pattern as the two-qubit version:
          - For each is_odd in contraction_array:
              * if is_odd, store last matrix as 'odd_exp', reduce step_exps by 1,
                multiply pairs, then re-append a single product with odd_exp
              * else, just multiply pairs
        Returns => final operator (squeezed).
        """
        step_exps = self.exponentials(ctrl_a, ctrl_b, ctrl_c)

        for is_odd in self.contraction_array:
            if is_odd:
                # Store the last exponential
                odd_exp = step_exps[-1]
                step_exps = step_exps[:-1]
                # Multiply pairs: step_exps[1::2] x step_exps[0:-1:2]
                merged = jnp.matmul(step_exps[1::2], step_exps[0:-1:2])
                # Multiply odd_exp with the last result
                new_tail = jnp.matmul(odd_exp, merged[-1])
                step_exps = jnp.concatenate([merged[:-1], new_tail[None,:,:]], axis=0)
            else:
                # Even case: multiply pairs [1::2] x [0::2]
                step_exps = jnp.matmul(step_exps[1::2], step_exps[0::2])

        return jnp.squeeze(step_exps)

    def metrics(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Compute performance metrics:
         - Infidelity between the final state and target state.
         - An adiabaticity measure based on a sub-block of the propagator.
        """
        VL_propagator = self.propagate(ctrl_a, ctrl_b, ctrl_c)
        psi_0 = self.psi_0
        psi_t = self.psi_t
        U_prop = VL_propagator[0:self.dim, 0:self.dim]
        final_state = jnp.matmul(U_prop, psi_0)
        overlap = jnp.matmul(jnp.conj(psi_t).T, final_state)
        infidelity = 1 - jnp.real(jnp.conj(overlap) * overlap)
        ad_integral = VL_propagator[0:self.dim, self.dim:2*self.dim]
        inner = jnp.matmul(ad_integral, psi_0)
        ad_metric = 1 - (1 / self.duration) * jnp.matmul(jnp.conj(final_state).T, inner)
        adiabaticity = jnp.real(ad_metric)
        return infidelity, adiabaticity

    def target(self, ctrl_a: jnp.ndarray, ctrl_b: jnp.ndarray, ctrl_c: jnp.ndarray):
        """
        Compute the weighted cost function:
         0.6 * infidelity + 0.4 * adiabaticity.
        """
        infid, adiab = self.metrics(ctrl_a, ctrl_b, ctrl_c)
        return 0.6 * infid + 0.4 * adiab