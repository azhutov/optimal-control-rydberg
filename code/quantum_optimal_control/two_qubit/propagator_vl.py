import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit, vmap, lax
from functools import partial

from arc import Rubidium87
from qutip import *


# ----------------------------------------------------------------------
# Small utility replacements for QuTiP style operations
# ----------------------------------------------------------------------
def dag(mat: jnp.ndarray) -> jnp.ndarray:
    """Conjugate transpose (dagger)."""
    return jnp.conj(mat.T)

def tensor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Tensor (Kronecker) product in JAX."""
    return jnp.kron(a, b)

def basis(dim: int, idx: int) -> jnp.ndarray:
    """Mimic qutip.basis(dim, idx): unit vector of length 'dim' with 1 in 'idx'."""
    e = jnp.zeros((dim,), dtype=jnp.complex128)
    return e.at[idx].set(1.0)

def outer(ket: jnp.ndarray) -> jnp.ndarray:
    """|ket><ket| as a matrix."""
    return jnp.outer(ket, jnp.conj(ket))


# ----------------------------------------------------------------------
# A helper for generating a real-space Gaussian array (to replace scipy.signal)
# ----------------------------------------------------------------------
def gaussian_array(M: int, std: float) -> jnp.ndarray:
    """
    Generate a Gaussian of length M with 'std' controlling the width.
    This mimics scipy.signal.gaussian(M, std).
    """
    x = jnp.arange(M, dtype=jnp.float64)
    c = 0.5 * (M - 1)
    return jnp.exp(-0.5 * ((x - c) / (std))**2) # match scipy definition: std / 2.0 => std https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html


# ----------------------------------------------------------------------
# Updated Van Loan Propagator in JAX
# ----------------------------------------------------------------------
class PropagatorVL:
    def __init__(
        self,
        input_dim: int,
        no_of_steps: int,
        pad: int,
        f_std: float,
        delta_t: float,
        del_total: float,
        V_int: float,
        Delta_i: float,
        Rabi_i: float,
        Rabi_r: float,
        Gammas_all: list
    ):
        """
        A JAX-based two-qubit "Van Loan" propagator that parallels the TF code.
        Differences vs. the older version:
          - Includes non-zero LP1..LP4 blocks (like the TF approach),
          - Adds them into the exponent via time-dependent controls,
          - Uses the same 'odd/even' multiplication pattern in propagate().

        Args:
            input_dim: Number of Gaussian basis modes for each control
            no_of_steps: Number of time points (excluding padding)
            pad: Number of padding points on each side
            f_std: Frequency-domain filter width for the pulses
            delta_t: Time step
            del_total: Additional detuning term
            V_int: Interaction strength
            Delta_i: Baseline single-atom detuning
            Rabi_i, Rabi_r: Baseline Rabi frequencies
            Gammas_all: List of decay rates [Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd]
        """
        self.dim = 4
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps
        self.padding = pad
        self.delta_t = delta_t
        self.del_total = del_total
        self.duration = (no_of_steps + 2 * pad) * delta_t

        self.Vint = V_int
        self.Rabi_i_val = Rabi_i
        self.Rabi_r_val = Rabi_r
        self.Delta_i_val = Delta_i
        self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd = Gammas_all

        # Precompute a frequency-domain Gaussian filter
        M = no_of_steps + 2 * pad
        freqs = jnp.fft.fftfreq(no_of_steps, d=delta_t)
        dfreq = freqs[1] - freqs[0]
        std_filter = f_std / dfreq
        filter_vals = gaussian_array(M, std_filter)
        self.Gaussian_filter = filter_vals.astype(jnp.complex128)

        # Create single-atom basis states (g_0, g_1, i, r)
        g_0 = basis(self.dim, 0)
        g_1 = basis(self.dim, 1)
        i_st = basis(self.dim, 2)
        r_st = basis(self.dim, 3)

        # Store some typical states for cost function
        self.psi_10 = jnp.kron(g_1, g_0)
        self.psi_11 = jnp.kron(g_1, g_1)
        self.psi_01 = jnp.kron(g_0, g_1)
        self.psi_pp = 0.5 * (
            tensor(g_0, g_0) + tensor(g_0, g_1)
            + tensor(g_1, g_0) + tensor(g_1, g_1)
        )

        # The target operator: +|00><00| +|01><01| +|10><10| -|11><11|
        # in 16x16 form
        U_t = (
            outer(tensor(g_0, g_0))
            + outer(tensor(g_0, g_1))
            + outer(tensor(g_1, g_0))
            - outer(tensor(g_1, g_1))
        )
        self.U_target = U_t

        # Build the block-lifted Hamiltonian + projector operators
        Hams = self._Hamiltonian(
            g_0, g_1, i_st, r_st,
            del_total, V_int,
            Delta_i, Rabi_i, Rabi_r,
            self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd
        )
        self.VL_Generators_ansatz = self._VL_Generators(Hams, g_0, g_1)

        # Build the "contraction array" for the same odd/even recursion:
        self.contraction_array = self.gen_contraction_array(M)

    # ------------------
    # Recursively define the "odd/even" contraction steps (like TF code)
    # ------------------
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

    # ----------------------------------------------------------------------
    # Build the underlying single/two-atom Hamiltonians
    # ----------------------------------------------------------------------
    @staticmethod
    def _Hamiltonian(
        g_0, g_1, i_st, r_st,
        del_total, V_int,
        Delta_i, Rabi_i, Rabi_r,
        Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd
    ):
        """Return a list [H_TQ_0, H_TQ_del_total, H_TQ_Delta, H_TQ_Rabi_i_re, ...]."""
        sig_00 = outer(g_0)
        sig_11 = outer(g_1)
        sig_ii = outer(i_st)
        sig_rr = outer(r_st)
        sig_1i = jnp.outer(g_1, jnp.conj(i_st))  # |1><i|
        sig_ir = jnp.outer(i_st, jnp.conj(r_st)) # |i><r|

        # Non-Hermitian decays, single-atom
        H_SQ_decay = (
            -1j * Gamma_i1 / 2.0 * sig_ii
            -1j * Gamma_10 / 2.0 * sig_11
            -1j * (Gamma_ri + Gamma_rd) / 2.0 * sig_rr
        )
        H_SQ_del = - del_total * sig_rr
        H_SQ_Delta_i = - Delta_i * sig_ii

        # Rabi re/im splitting
        # re_part( X ) = -0.5*(X + X^\dagger), im_part( X ) = -0.5j*(X - X^\dagger)
        def re_part(operator):
            return -0.5 * (operator + dag(operator))
        def im_part(operator):
            return -0.5j * (operator - dag(operator))

        H_SQ_Rabi_i_re = re_part(Rabi_i * sig_1i)
        H_SQ_Rabi_i_im = im_part(Rabi_i * sig_1i)
        H_SQ_Rabi_r_re = re_part(Rabi_r * sig_ir)
        H_SQ_Rabi_r_im = im_part(Rabi_r * sig_ir)

        dim = 4
        I = jnp.eye(dim, dtype=jnp.complex128)
        def t2(a, b):
            return jnp.kron(a, b)

        # Two-atom expansions:
        H_TQ_0 = t2(I, H_SQ_decay) + t2(H_SQ_decay, I) + V_int * t2(sig_rr, sig_rr)
        H_TQ_del_total = t2(I, H_SQ_del) + t2(H_SQ_del, I)
        H_TQ_Delta = t2(I, H_SQ_Delta_i) + t2(H_SQ_Delta_i, I)
        H_TQ_Rabi_i_re = t2(H_SQ_Rabi_i_re, I) + t2(I, H_SQ_Rabi_i_re)
        H_TQ_Rabi_i_im = t2(H_SQ_Rabi_i_im, I) + t2(I, H_SQ_Rabi_i_im)
        H_TQ_Rabi_r_re = t2(H_SQ_Rabi_r_re, I) + t2(I, H_SQ_Rabi_r_re)
        H_TQ_Rabi_r_im = t2(H_SQ_Rabi_r_im, I) + t2(I, H_SQ_Rabi_r_im)

        return [
            H_TQ_0,
            H_TQ_del_total,
            H_TQ_Delta,
            H_TQ_Rabi_i_re,
            H_TQ_Rabi_i_im,
            H_TQ_Rabi_r_re,
            H_TQ_Rabi_r_im,
        ]

    @staticmethod
    def _VL_Generators(Hams, g_0, g_1):
        """
        Construct a list of 11 block-lifted 32x32 operators:
          0: H0
          1: H1
          2: H2
          3: H3_real
          4: H3_im
          5: H4_real
          6: H4_im
          7..10: LP1..LP4  (non-zero in top-right block)

        Mirroring the TF code, we do:
           LU_0 = [[H0, 0],[0,H0]] etc.
           ...
           LP1 = [[0, TENSOR(...)] ,[0,0]]
           ...
        """
        dim = 4
        dim_sq = dim*dim
        Zero16 = jnp.zeros((dim_sq, dim_sq), dtype=jnp.complex128)

        def block2x2(a, b, c, d):
            """Construct a 2x2 block matrix via jnp.block."""
            return jnp.block([[a, b],
                              [c, d]])

        H0, Hdel_tot, HDelta, Hi_re, Hi_im, Hr_re, Hr_im = Hams

        # 2x2 block expansions
        LU0 = block2x2(H0, Zero16, Zero16, H0)
        LU1 = block2x2(Hdel_tot, Zero16, Zero16, Hdel_tot)
        LU2 = block2x2(HDelta, Zero16, Zero16, HDelta)
        LU3r = block2x2(Hi_re, Zero16, Zero16, Hi_re)
        LU3i = block2x2(Hi_im, Zero16, Zero16, Hi_im)
        LU4r = block2x2(Hr_re, Zero16, Zero16, Hr_re)
        LU4i = block2x2(Hr_im, Zero16, Zero16, Hr_im)

        # Now the 4 "LP" blocks in top-right block only
        # as in the TF code: LP1 = block([[0, tensor(g_1*g_1^dag, g_0*g_0^dag)], [0,0]])
        # We'll do the same in JAX style
        # g_1*g_1.dag() => outer(g_1), g_0*g_0.dag() => outer(g_0)
        LP1_16 = tensor(outer(g_1), outer(g_0))
        LP2_16 = tensor(outer(g_0), outer(g_0)) * 0.0  # replace as needed, or replicate your TF logic
        LP3_16 = tensor(outer(g_0), outer(g_0)) * 0.0
        LP4_16 = tensor(outer(g_0), outer(g_0)) * 0.0

        # If you truly want the same exact projectors as your TF code:
        #   LP1 = [Zero, (g_1*g_1.dag() ⊗ g_0*g_0.dag())],
        #   LP2 = [Zero, (r*r.dag()   ⊗ g_0*g_0.dag())],
        #   LP3 = [Zero, (g_1*r.dag() ⊗ g_0*g_0.dag())],
        #   LP4 = [Zero, (r*g_1.dag() ⊗ g_0*g_0.dag())].
        # 
        # Here is an exact replication of the TF snippet:
        #   LP1 = np.block([
        #       [Zeroes, tensor(g_1*g_1.dag(), g_0*g_0.dag())],
        #       [Zeroes, Zeroes]
        #   ])
        #   LP2 = np.block([
        #       [Zeroes, tensor(r*r.dag(), g_0*g_0.dag())],
        #       [Zeroes, Zeroes]
        #   ])
        #   LP3 = np.block([
        #       [Zeroes, tensor(g_1*r.dag(), g_0*g_0.dag())],
        #       [Zeroes, Zeroes]
        #   ])
        #   LP4 = np.block([
        #       [Zeroes, tensor(r*g_1.dag(), g_0*g_0.dag())],
        #       [Zeroes, Zeroes]
        #   ])
        # 
        # For illustration, below is a direct JAX version. If you have i, r, etc.
        # states already, define them above or adapt as needed. We'll assume
        # g_1, g_0, i_st, r_st exist and do the same approach:

        # Let's define them quickly here for clarity
        # (In the real code, you'd keep i_st,r_st from above if needed.)
        # NOTE: This snippet exactly matches the TF "LP" usage:
        #   LP1 => (|1><1|⊗|0><0|) in top-right
        #   LP2 => (|r><r|⊗|0><0|) in top-right
        #   LP3 => (|1><r|⊗|0><0|) in top-right
        #   LP4 => (|r><1|⊗|0><0|) in top-right
        #   but we need the actual g_1, r_st in scope. Let's assume you have them:
        r_st = basis(dim, 3)
        LP1_16 = tensor(outer(g_1), outer(g_0))
        LP2_16 = tensor(outer(r_st), outer(g_0))
        LP3_16 = tensor(jnp.outer(g_1, jnp.conj(r_st)), outer(g_0))
        LP4_16 = tensor(jnp.outer(r_st, jnp.conj(g_1)), outer(g_0))

        LP1 = block2x2(Zero16, LP1_16, Zero16, Zero16)
        LP2 = block2x2(Zero16, LP2_16, Zero16, Zero16)
        LP3 = block2x2(Zero16, LP3_16, Zero16, Zero16)
        LP4 = block2x2(Zero16, LP4_16, Zero16, Zero16)

        return jnp.stack([LU0, LU1, LU2, LU3r, LU3i, LU4r, LU4i, LP1, LP2, LP3, LP4], axis=0)

    # ----------------------------------------------------------------------
    # Convert from (ctrl_a, ctrl_b, ctrl_c) => physically-bounded waveforms => filter
    # ----------------------------------------------------------------------
    def _gen_basis_matrix(self, b_vals: jnp.ndarray, c_vals: jnp.ndarray):
        """
        Creates time-domain Gaussian modes: exp(-((t - b)^2 / c^2)), with t in [-1,1].
        Returns shape [no_of_steps, input_dim].
        """
        # We'll define a times array in [-1, 1] (like TF)
        times = jnp.linspace(-1.0, 1.0, self.no_of_steps, dtype=jnp.float64)

        def single_mode(params):
            b_val, c_val = params
            return jnp.exp(-((times - b_val)**2) / (c_val**2))

        bc = jnp.stack([b_vals, c_vals], axis=1)  # shape [input_dim, 2]
        modes = vmap(single_mode, in_axes=0, out_axes=0)(bc)  # [input_dim, no_of_steps]
        return modes.T  # => [no_of_steps, input_dim]

    def _transform_amplitudes(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Summation of Gaussian basis modes => raw waveforms (like TF code).
        shape: each ctrl_* is [input_dim, 5].
        Returns => [no_of_steps, 5].
        """
        num_ctrls = ctrl_a.shape[1]
        wave_list = []
        for j in range(num_ctrls):
            # clamp b,c with tanh
            b_clamped = jnp.tanh(ctrl_b[:, j])
            c_clamped = jnp.tanh(ctrl_c[:, j])

            basis_mat = self._gen_basis_matrix(b_clamped, c_clamped)
            # multiply by amplitude column => [input_dim]
            amp_col = ctrl_a[:, j]
            wave_list.append(basis_mat @ amp_col)  # => shape [no_of_steps]

        return jnp.stack(wave_list, axis=1)

    def _regularize_amplitudes(self, ctrl_a, ctrl_b, ctrl_c):
        """
        - Delta_i(t) => forced to -1.0 (like your TF snippet),
        - Rabi_i and Rabi_r => [Amplitude, Phase] with 1 - exp(-(...^2)) for amplitude,
          tanh(...) for phase.
        Returns => shape [no_of_steps, 5].
        """
        ta = self._transform_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # => [no_of_steps, 5]

        Delta_col = 0.0 * ta[:, 0:1] - 1.0
        Rabi_i_mag = 1.0 - jnp.exp(-jnp.square(ta[:, 1:2]))
        Rabi_i_phase = jnp.tanh(ta[:, 2:3])
        Rabi_r_mag = 1.0 - jnp.exp(-jnp.square(ta[:, 3:4]))
        Rabi_r_phase = jnp.tanh(ta[:, 4:5])

        return jnp.concatenate(
            [Delta_col, Rabi_i_mag, Rabi_i_phase, Rabi_r_mag, Rabi_r_phase],
            axis=1
        )

    def _filter_amplitudes(self, amplitudes: jnp.ndarray):
        """
        Frequency filter each channel.  amplitudes: [no_of_steps, 5].
        Then pad => [M, 5], do FFT, multiply by self.Gaussian_filter, iFFT.
        Returns => shape [M, 5], where M=no_of_steps+2*pad.
        """
        pad = self.padding
        M = self.no_of_steps + 2*pad

        padded = jnp.pad(
            amplitudes.astype(jnp.complex128),
            ((pad, pad),(0,0)),
            mode='constant'
        )

        def do_filter(col: jnp.ndarray):
            freq_amp = jnp.fft.fftshift(jnp.fft.fft(col))
            filtered = self.Gaussian_filter * freq_amp
            return jnp.fft.ifft(jnp.fft.ifftshift(filtered))

        padded_t = padded.T  # [5, M]
        filtered_t = vmap(do_filter, in_axes=0, out_axes=0)(padded_t)  # => [5, M]
        return jnp.real(filtered_t).T  # => [M, 5]

    def return_physical_amplitudes(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Combine all steps => filtered waveforms in time. [M, 5].
        """
        raw_reg = self._regularize_amplitudes(ctrl_a, ctrl_b, ctrl_c)
        return self._filter_amplitudes(raw_reg)

    def return_auxiliary_amplitudes(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Create the 9 columns used to multiply the last 9 blocks of the 32x32 matrix.
          - 1st 5 columns => drift-like Hamiltonians (Delta, Rabi_i re/im, Rabi_r re/im)
            multiplied by -1j to match TF code,
          - Last 4 columns => "LP1..LP4" projector terms.
        Output => shape [M, 9].
        """
        wave = self.return_physical_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # => [M,5]

        # Convert amplitude+phase => complex Rabi
        # Rabi_i
        rabi_i_mag = self.Rabi_i_val * wave[:,1]
        rabi_i_phase = jnp.pi * wave[:,2]
        rabi_i_cplx = rabi_i_mag * jnp.exp(1j * rabi_i_phase)

        # Rabi_r
        rabi_r_mag = self.Rabi_r_val * wave[:,3]
        rabi_r_phase = jnp.pi * wave[:,4]
        rabi_r_cplx = rabi_r_mag * jnp.exp(1j * rabi_r_phase)

        # Projector denominators
        LP_1 = rabi_r_cplx * jnp.conj(rabi_r_cplx)  # |rabi_r|^2
        LP_2 = rabi_i_cplx * jnp.conj(rabi_i_cplx)  # |rabi_i|^2
        LP_3 = -(rabi_i_cplx * rabi_r_cplx)
        LP_4 = -(jnp.conj(rabi_i_cplx) * jnp.conj(rabi_r_cplx))
        norm_denom = LP_1 + LP_2
        eps = 1e-12
        norm_denom_safe = jnp.where(norm_denom < eps, eps, norm_denom)
        P_stack = jnp.stack([
            LP_1 / norm_denom_safe,
            LP_2 / norm_denom_safe,
            LP_3 / norm_denom_safe,
            LP_4 / norm_denom_safe
        ], axis=1)  # [M,4]

        # For the "Hamiltonian" columns: (Delta, Rabi_i re, Rabi_i im, Rabi_r re, Rabi_r im)
        # wave[:,0] => Delta (unscaled, which is ~-1),
        # real(rabi_i_cplx)/Rabi_i_val, imag(rabi_i_cplx)/Rabi_i_val, ...
        Delta_col = wave[:,0]
        Ri_re = jnp.real(rabi_i_cplx)/self.Rabi_i_val
        Ri_im = jnp.imag(rabi_i_cplx)/self.Rabi_i_val
        Rr_re = jnp.real(rabi_r_cplx)/self.Rabi_r_val
        Rr_im = jnp.imag(rabi_r_cplx)/self.Rabi_r_val

        # Multiply by (-1j) to match TF:
        U_stack = jnp.stack([Delta_col, Ri_re, Ri_im, Rr_re, Rr_im], axis=1).astype(jnp.complex128)
        U_stack *= -1j  # shape [M,5]

        return jnp.concatenate([U_stack, P_stack], axis=1)  # => [M,9]

    # ----------------------------------------------------------------------
    # Build exponentials for each time step
    # ----------------------------------------------------------------------
    def exponentials(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Summation of drift blocks (gens[0], gens[1]) plus
        time-dependent blocks (gens[2..10]) * "auxiliary_amplitudes".
        Return => shape [M,32,32].
        """
        gens = self.VL_Generators_ansatz  # shape [11, 32,32]
        aux = self.return_auxiliary_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # [M, 9]

        # drift = -1j*(gens[0] + gens[1])
        drift = -1j * (gens[0] + gens[1])  # shape [32,32]

        # The rest => gens[2..] => shape [9, 32,32], multiply by aux => shape [M,9]
        # => sum => shape [M,32,32]
        control_part = jnp.tensordot(aux, gens[2:], axes=[[1],[0]])  # => [M, 32,32]

        exponent = self.delta_t * (drift[None,:,:] + control_part)  # => [M,32,32]

        # Batch-exponentiate each [32,32] slice
        def single_expm(mat):
            return jla.expm(mat)

        # vmap across the M dimension => we want the result also shape [M,32,32]
        step_exps = vmap(single_expm, in_axes=0, out_axes=0)(exponent)  # => [M,32,32]
        return step_exps

    # ----------------------------------------------------------------------
    # Replicate the "odd/even" recursion from TF code
    # ----------------------------------------------------------------------
    def propagate(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Follows the same pattern as the TF version:
          - step_exps = exponentials() => shape [M, 32,32]
          - for is_odd in self.contraction_array:
              * if is_odd, store last matrix as 'odd_exp', reduce step_exps by 1,
                multiply pairs, then re-append a single product with odd_exp
              * else, just multiply pairs
        Returns => final 32x32 operator (squeezed).
        """
        step_exps = self.exponentials(ctrl_a, ctrl_b, ctrl_c)  # [M,32,32]

        # We'll do a Python loop to mirror your TF approach.
        # This won't be fully JIT-compiled, but it preserves the same logic.
        for is_odd in self.contraction_array:
            if is_odd:
                # separate out the "odd" last slice
                odd_exp = step_exps[-1]               # shape [32,32]
                step_exps = step_exps[:-1]            # [M-1, 32,32]
                # multiply pairs of them: step_exps[1::2] x step_exps[0:-1:2]
                merged = jnp.matmul(step_exps[1::2], step_exps[0:-1:2])
                # shape => [floor((M-1)/2), 32,32]

                # then we replace step_exps with everything except last,
                # plus matmul(odd_exp, last)
                # but note the TF code does:
                #   step_exps = tf.concat([ step_exps[0:-1], [ odd_exp * step_exps[-1] ] ], axis=0)
                #   except we've replaced step_exps with 'merged' now.
                # The direct TF snippet is:
                #   step_exps = tf.linalg.matmul(step_exps[1::2], step_exps[0:-1:2])
                #   step_exps = tf.concat([step_exps[0:-1,:,:],
                #                         [tf.linalg.matmul(odd_exp, step_exps[-1,:,:])]], 0)
                # We'll do the same pattern:
                #   merged => shape [X,32,32], we do merged[:-1] then multiply(odd_exp, merged[-1])
                new_tail = jnp.matmul(odd_exp, merged[-1])
                step_exps = jnp.concatenate(
                    [merged[:-1], new_tail[None,:,:]],
                    axis=0
                )
            else:
                # even
                # step_exps => multiply pairs: [1::2] x [0::2]
                step_exps = jnp.matmul(step_exps[1::2], step_exps[0::2])
                # shape => [floor(M/2), 32,32]

        # final => should be shape [1,32,32], so squeeze => [32,32]
        return step_exps.squeeze()

    # ----------------------------------------------------------------------
    # Evaluate cost function pieces
    # ----------------------------------------------------------------------
    def metrics(self, ctrl_a, ctrl_b, ctrl_c):
        """
        1) Run propagate() => final 32x32
        2) Extract top-left 16x16 => U_prop
        3) Compute infidelity vs U_target
        4) Extract [0:16, 16:32] => ad_integral, measure adiabaticity for |10>
        """
        VL_propagator = self.propagate(ctrl_a, ctrl_b, ctrl_c)  # => [32,32]

        dim_sq = self.dim * self.dim
        U_prop = VL_propagator[0:dim_sq, 0:dim_sq]
        U_target = self.U_target

        # Infidelity measure
        dtrace = jnp.trace(U_target.conj().T @ U_prop)
        dtrace_norm = jnp.trace(U_target.conj().T @ U_target)
        ratio = (dtrace * jnp.conj(dtrace)) / (dtrace_norm**2)
        infidelity = 1.0 - jnp.real(ratio)

        # Adiabatic metric: sub-block [0:16, 16:32]
        ad_integral = VL_propagator[0:dim_sq, dim_sq:2*dim_sq]
        final_state_10 = U_prop @ self.psi_10
        tmp = ad_integral @ self.psi_10
        overlap = jnp.conj(final_state_10).T @ tmp
        adiabaticity = 1.0 - (1.0/self.duration)*overlap
        adiabaticity = jnp.real(adiabaticity)

        return infidelity, adiabaticity

    def target(self, ctrl_a, ctrl_b, ctrl_c):
        """
        Weighted cost => 0.2 * infidelity + 0.8 * adiabaticity
        """
        infid, adiab = self.metrics(ctrl_a, ctrl_b, ctrl_c)
        return 0.2*infid + 0.8*adiab


# ----------------------------------------------------------------------
# A simple gradient-descent step in JAX
# ----------------------------------------------------------------------
@partial(jit, static_argnums=0)
def single_optimization_step(propagator: PropagatorVL,
                            ctrl_a: jnp.ndarray,
                            ctrl_b: jnp.ndarray,
                            ctrl_c: jnp.ndarray,
                            lr: float = 0.02):
    """
    Perform one gradient-based optimization iteration.
    """
    def loss_fn(a, b, c):
        return propagator.target(a, b, c)

    cost, grads = jax.value_and_grad(loss_fn, argnums=(0,1,2))(ctrl_a, ctrl_b, ctrl_c)
    gradA, gradB, gradC = grads

    # gradient update
    new_ctrl_a = ctrl_a - lr * gradA
    new_ctrl_b = ctrl_b - lr * gradB
    new_ctrl_c = ctrl_c - lr * gradC

    return cost, new_ctrl_a, new_ctrl_b, new_ctrl_c
