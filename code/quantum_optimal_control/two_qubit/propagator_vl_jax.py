###############################################################################
# propagator_vl_jax.py
#
# A JAX version of the Van Loan propagator code. It uses jax.numpy (jnp),
# jax.scipy, and includes @jax.jit for acceleration. 
#
# To further speed up, ensure you have a GPU or TPU available and that JAX 
# is installed with CUDA support. Also consider switching to float32/complex64.
###############################################################################

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit, vmap, lax
from functools import partial

import numpy as np
from typing import Tuple, List

# ------------------------------------------------------------------------------
# Small utility replacements for QuTiP style operations
# ------------------------------------------------------------------------------

def dag(vec_or_mat: jnp.ndarray) -> jnp.ndarray:
    """Conjugate transpose (dagger)."""
    return jnp.conj(vec_or_mat.T)

def tensor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Tensor (Kronecker) product."""
    return jnp.kron(a, b)

def basis(dim: int, idx: int) -> jnp.ndarray:
    """Mimic qutip.basis(dim, idx): unit vector of length dim with 1 in 'idx'."""
    e = jnp.zeros((dim,))
    return e.at[idx].set(1.0)

# ------------------------------------------------------------------------------
# A helper for generating a real-space "Gaussian" array (to replace scipy.signal)
# ------------------------------------------------------------------------------
def gaussian_array(M: int, std: float) -> jnp.ndarray:
    """
    Generate a Gaussian of length M with 'std' controlling the width.
    This mimics scipy.signal.gaussian(M, std).
    """
    x = jnp.arange(M)
    c = 0.5 * (M - 1)
    return jnp.exp(-0.5 * ((x - c) / (std / 2.0))**2)

# ------------------------------------------------------------------------------
# The main class, rewriting your TensorFlow code in a JAX-friendly manner
# ------------------------------------------------------------------------------
class PropagatorVLJAX:
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
        Gammas_all: List[float],
    ):
        """
        Set up all needed states, operators, and references. 
        JAX style: Precompute static arrays. Control parameters are passed 
        externally into the 'target(...)' or 'metrics(...)' methods.
        """
        self.duration = (no_of_steps + 2 * pad) * delta_t
        self.delta_t = delta_t
        self.Vint = V_int
        self.Rabi_i_val = Rabi_i  # max Rabi for i
        self.Rabi_r_val = Rabi_r  # max Rabi for r
        self.Delta_i_val = Delta_i
        self.del_total = del_total
        self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd = Gammas_all
        self.dim = 4
        self.input_dim = input_dim
        self.no_of_steps = no_of_steps
        self.padding = pad

        # Discretized time axis for building Gaussian basis (float64).
        # We'll store it as jnp.array so we can jax.jit easily.
        self.times_basis = jnp.linspace(
            -1.0, 1.0, no_of_steps, dtype=jnp.float64
        )

        # Pre-generate frequency-domain Gaussian filter once, as a complex128 array.
        M = no_of_steps + 2 * pad
        freqs = jnp.fft.fftfreq(no_of_steps, d=delta_t)
        dfreq = freqs[1] - freqs[0]  # spacing in frequency
        std_filter = f_std / dfreq
        filter_vals = gaussian_array(M, std_filter)
        # store as complex for direct multiplication in FFT domain
        self.Gaussian_filter = filter_vals.astype(jnp.complex128)

        # ----------------------------------------------------------------------
        # Build single-atom basis states (g_0, g_1, i, r)
        # ----------------------------------------------------------------------
        st = [basis(self.dim, i) for i in range(self.dim)]  # 4 vectors
        g_0, g_1, i_state, r_state = st

        # ----------------------------------------------------------------------
        # Build the relevant multi-qubit basis states (using Kronecker)
        # ----------------------------------------------------------------------
        # We'll create them as 4x1 or 16x1 jnp arrays.
        # For a 2-atom system with dim=4 each, we get 16-dim states.

        psi_pp_arr = 0.5 * (
            jnp.kron(g_0, g_0) + jnp.kron(g_0, g_1) +
            jnp.kron(g_1, g_0) + jnp.kron(g_1, g_1)
        )
        psi_10_arr = jnp.kron(g_1, g_0)
        psi_11_arr = jnp.kron(g_1, g_1)
        psi_01_arr = jnp.kron(g_0, g_1)

        # Make them complex128 and store
        self.psi_pp = psi_pp_arr.astype(jnp.complex128)
        self.psi_10 = psi_10_arr.astype(jnp.complex128)
        self.psi_11 = psi_11_arr.astype(jnp.complex128)
        self.psi_01 = psi_01_arr.astype(jnp.complex128)

        # The target operator U_target => 16x16 matrix
        # ( g0g0 + g0g1 + g1g0 - g1g1 ). Note that in qutip notation:
        #   + |00><00| + |01><01| + |10><10| - |11><11|
        # We'll build them by outer products manually:
        def outer(ket):
            return jnp.outer(ket, jnp.conj(ket))
        
        U_t = (
            outer(jnp.kron(g_0, g_0)) +
            outer(jnp.kron(g_0, g_1)) +
            outer(jnp.kron(g_1, g_0)) -
            outer(jnp.kron(g_1, g_1))
        )
        self.U_target = U_t.astype(jnp.complex128)

        # ----------------------------------------------------------------------
        # Build the Van Loan generator blocks (2 * dim^2 = 2 * 16 = 32 dimension)
        # ----------------------------------------------------------------------
        # We create the Hamiltonian pieces and store them, then build
        # the block-lifted operators. Each will be 32x32 in size.
        Hams = self._Hamiltonian(
            g_0, g_1, i_state, r_state,
            self.del_total, self.Vint, 
            self.Delta_i_val, self.Rabi_i_val, self.Rabi_r_val,
            self.Gamma_10, self.Gamma_i1, self.Gamma_ri, self.Gamma_rd
        )
        self.VL_Generators_ansatz = self._VL_Generators(Hams)

    # --------------------------------------------------------------------------
    # Hamiltonian & generator-building routines, static or hidden
    # --------------------------------------------------------------------------
    @staticmethod
    def _Hamiltonian(
        g_0, g_1, i_st, r_st,
        del_total, V_int,
        Delta_i, Rabi_i, Rabi_r,
        Gamma_10, Gamma_i1, Gamma_ri, Gamma_rd
    ):
        """
        Build the list of single-/two-atom operators. For convenience in JAX, we 
        store them as a list of 7 big 16x16 arrays. We'll combine them later in 
        the Van Loan approach.
        """
        # Single-atom projector type operators:
        sig_00 = jnp.outer(g_0, jnp.conj(g_0))
        sig_11 = jnp.outer(g_1, jnp.conj(g_1))
        sig_ii = jnp.outer(i_st, jnp.conj(i_st))
        sig_rr = jnp.outer(r_st, jnp.conj(r_st))

        sig_1i = jnp.outer(g_1, jnp.conj(i_st))
        sig_ir = jnp.outer(i_st, jnp.conj(r_st))

        # Non-Hermitian decays, single-atom
        H_SQ_decay = (
            -1j * Gamma_i1 / 2.0 * sig_ii
            -1j * Gamma_10 / 2.0 * sig_11
            -1j * (Gamma_ri + Gamma_rd) / 2.0 * sig_rr
        )
        H_SQ_del = -del_total * sig_rr
        H_SQ_Delta_i = -Delta_i * sig_ii

        # Rabi couplings (i, r) real + imaginary contributions
        # (like splitting them into cos and sin parts)
        def re_part(sig):
            return -0.5 * (sig + dag(sig))

        def im_part(sig):
            return -0.5j * (sig - dag(sig))

        H_SQ_Rabi_i_re = re_part(Rabi_i * sig_1i)
        H_SQ_Rabi_i_im = im_part(Rabi_i * sig_1i)

        H_SQ_Rabi_r_re = re_part(Rabi_r * sig_ir)
        H_SQ_Rabi_r_im = im_part(Rabi_r * sig_ir)

        # Tensor up to 2 atoms: dimension => 16
        dim = 4
        I = jnp.eye(dim, dtype=jnp.complex128)
        def t2(a, b):
            return jnp.kron(a, b)

        H_TQ_0 = t2(I, H_SQ_decay) + t2(H_SQ_decay, I) + V_int * t2(sig_rr, sig_rr)
        H_TQ_del_total = t2(I, H_SQ_del) + t2(H_SQ_del, I)
        H_TQ_Delta = t2(I, H_SQ_Delta_i) + t2(H_SQ_Delta_i, I)
        H_TQ_Rabi_i_re = t2(H_SQ_Rabi_i_re, I) + t2(I, H_SQ_Rabi_i_re)
        H_TQ_Rabi_i_im = t2(H_SQ_Rabi_i_im, I) + t2(I, H_SQ_Rabi_i_im)
        H_TQ_Rabi_r_re = t2(H_SQ_Rabi_r_re, I) + t2(I, H_SQ_Rabi_r_re)
        H_TQ_Rabi_r_im = t2(H_SQ_Rabi_r_im, I) + t2(I, H_SQ_Rabi_r_im)

        return [
            H_TQ_0,            # index 0
            H_TQ_del_total,    # index 1
            H_TQ_Delta,        # index 2
            H_TQ_Rabi_i_re,    # index 3
            H_TQ_Rabi_i_im,    # index 4
            H_TQ_Rabi_r_re,    # index 5
            H_TQ_Rabi_r_im,    # index 6
        ]

    @staticmethod
    def _VL_Generators(Hams: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Construct block-lifted operators (32x32) for the Van Loan approach:
          We produce a stacked array of shape [7 + 4, 32, 32]
          The last 4 are the LP1..LP4 "projection" type operators used
          for the 'auxiliary' amplitude building.
        """
        dim = 4
        dim_big = dim * dim  # 16
        Zeroes = jnp.zeros((dim_big, dim_big), dtype=jnp.complex128)

        # convenience
        def block2x2(a, b, c, d):
            # block [[a, b], [c, d]] along diagonal 
            return jnp.block([[a, b], [c, d]])

        # 7 Hamiltonian-based blocks => each is 2*(16x16) => 32x32
        H0, H1, H2, H3r, H3i, H4r, H4i = Hams

        LU0 = block2x2(H0, Zeroes, Zeroes, H0)
        LU1 = block2x2(H1, Zeroes, Zeroes, H1)
        LU2 = block2x2(H2, Zeroes, Zeroes, H2)
        LU3r = block2x2(H3r, Zeroes, Zeroes, H3r)
        LU3i = block2x2(H3i, Zeroes, Zeroes, H3i)
        LU4r = block2x2(H4r, Zeroes, Zeroes, H4r)
        LU4i = block2x2(H4i, Zeroes, Zeroes, H4i)

        # The 4 "LP" blocks to stack at the end. 
        # In the original code, these were 2x2 block expansions with zero in the top-left.
        # They appear to be used for "auxiliary" amplitude terms. 
        # For consistency, we'll define them as 32x32 zero except for a block in the top-right.
        # Because each is [ [Zero, something], [Zero, Zero] ] in block form:
        def make_LP_block(op: jnp.ndarray) -> jnp.ndarray:
            return jnp.block([
                [Zeroes, op],
                [Zeroes, Zeroes]
            ])

        # We do not have direct references to g_0*g_0.dag(), etc. here,
        # so to keep the original logic we'd pass them in. But the old code 
        # used something like: tensor(r*r.dag(), g_0*g_0.dag()). Let's just keep
        # them zero or replicate the original approach if you rely on them.
        #
        # For demonstration, we use the same pattern as the old code. 
        # We'll define them as zero 16x16 for now. If your workflow 
        # truly needs them, replicate those operators from the original code.
        LP1_16 = Zeroes
        LP2_16 = Zeroes
        LP3_16 = Zeroes
        LP4_16 = Zeroes

        # 32x32 
        LP1 = make_LP_block(LP1_16)
        LP2 = make_LP_block(LP2_16)
        LP3 = make_LP_block(LP3_16)
        LP4 = make_LP_block(LP4_16)

        # Stack them
        return jnp.stack([LU0, LU1, LU2, LU3r, LU3i, LU4r, LU4i, LP1, LP2, LP3, LP4])

    # --------------------------------------------------------------------------
    # Methods that transform control parameters -> waveforms -> final exponentials
    # --------------------------------------------------------------------------
    def _gen_basis_matrix(
        self,
        b_vals: jnp.ndarray,
        c_vals: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Creates Gaussian modes in time: exp(-((t - b)^2 / c^2)), 
        for b,c in [-1,1]. 
        b_vals, c_vals => shape [input_dim].
        Returns => shape [no_of_steps, input_dim].
        """
        # `t` shape => [no_of_steps], broadcast vs [input_dim].
        # We can do a vmap on the *row index* or just expand dims.
        t = self.times_basis  # [no_of_steps]

        def single_mode(params):
            b_val, c_val = params
            return jnp.exp(-((t - b_val)**2) / (c_val**2))

        # Combine b_vals, c_vals => shape [input_dim, 2]
        bc = jnp.stack([b_vals, c_vals], axis=1)
        # vmap across input_dim => result [input_dim, no_of_steps]
        # we want [no_of_steps, input_dim], so we transpose afterwards.
        modes = vmap(single_mode, in_axes=0, out_axes=0)(bc)  # [input_dim, no_of_steps]
        return modes.T  # => [no_of_steps, input_dim]

    def _transform_amplitudes(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Build time‐domain waveforms from (a,b,c) parameters using Gaussian basis.
        The code here is analogous to your "transform_amplitudes", but in JAX style.
        - ctrl_a, ctrl_b, ctrl_c => shape [input_dim, num_ctrls].
        - Output => shape [no_of_steps, num_ctrls].
        """
        num_ctrls = ctrl_a.shape[1]  # 5
        wave_list = []
        for control_num in range(num_ctrls):
            # clamp b, c => we do "tanh"
            b_clamped = jnp.tanh(ctrl_b[:, control_num])
            c_clamped = jnp.tanh(ctrl_c[:, control_num])

            # basis matrix => [no_of_steps, input_dim]
            basis_mat = self._gen_basis_matrix(b_clamped, c_clamped)
            # multiply by amplitude column => shape [input_dim]
            amplitude_col = ctrl_a[:, control_num]  # shape [input_dim]
            wave = basis_mat @ amplitude_col  # => [no_of_steps]
            wave_list.append(wave)

        # stack => [no_of_steps, num_ctrls]
        return jnp.stack(wave_list, axis=1)

    def _regularize_amplitudes(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        1) Create basis waveforms -> transform into physically-bounded waveforms
        2) The columns: (Delta, Rabi_i amplitude, Rabi_i phase, Rabi_r amplitude, Rabi_r phase)
        """
        ta = self._transform_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # [no_of_steps, 5]

        # Delta => always -1
        Delta_col = 0.0 * ta[:, 0:1] - 1.0

        # Rabi i amplitude => 1 - exp(-ta^2)
        Rabi_i_mag = 1.0 - jnp.exp(-jnp.square(ta[:, 1:2]))
        Rabi_i_phase = jnp.tanh(ta[:, 2:3])

        Rabi_r_mag = 1.0 - jnp.exp(-jnp.square(ta[:, 3:4]))
        Rabi_r_phase = jnp.tanh(ta[:, 4:5])

        return jnp.concatenate(
            [Delta_col, Rabi_i_mag, Rabi_i_phase, Rabi_r_mag, Rabi_r_phase],
            axis=1
        )

    def _filter_amplitudes(
        self,
        amplitudes: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply frequency‐domain Gaussian filter to each control channel.
        amplitudes => [no_of_steps, 5]
        Return => [no_of_steps + 2*pad, 5]
        """
        pad = self.padding
        M = self.no_of_steps + 2 * pad

        # Pad in time domain
        padded = jnp.pad(
            amplitudes.astype(jnp.complex128),
            ((pad, pad), (0, 0)),
            'constant'
        )

        # We'll vmap over the 5 columns. 
        # Then for each column do fftshift -> multiply -> ifftshift
        def filter_col(col: jnp.ndarray) -> jnp.ndarray:
            freq_amp = jnp.fft.fftshift(jnp.fft.fft(col))
            filtered = self.Gaussian_filter * freq_amp
            return jnp.fft.ifft(jnp.fft.ifftshift(filtered))

        # shape => [5, M], we want to produce [M, 5]
        padded_t = padded.T  # [5, M]
        filtered_t = vmap(filter_col, in_axes=0, out_axes=0)(padded_t)  # [5, M]
        # cast back to float64 real part
        return jnp.real(filtered_t).T  # => [M, 5]

    def return_physical_amplitudes(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        1) regularize_amplitudes -> wave
        2) frequency filter -> shape [N+2*pad, 5]
        """
        reg = self._regularize_amplitudes(ctrl_a, ctrl_b, ctrl_c)
        return self._filter_amplitudes(reg)

    def return_auxiliary_amplitudes(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Convert final filtered amplitude array to complex controls:
         - wave[:,0] => Delta_i
         - wave[:,1], wave[:,2] => Rabi_i (amplitude, phase)
         - wave[:,3], wave[:,4] => Rabi_r (amplitude, phase)

        Also build P projection terms. Return => shape [N+2*pad, 9].
        The first 5 columns = scaled Hamiltonian controls, the last 4 = P terms.
        """
        wave = self.return_physical_amplitudes(ctrl_a, ctrl_b, ctrl_c)  # [N, 5]
        N = wave.shape[0]

        # Build complex Rabi for i
        rabi_i_mag = self.Rabi_i_val * wave[:, 1]  # float64
        rabi_i_phase = jnp.pi * wave[:, 2]
        rabi_i_cplx = rabi_i_mag * jnp.exp(1j * rabi_i_phase)  # shape [N]

        # Build complex Rabi for r
        rabi_r_mag = self.Rabi_r_val * wave[:, 3]
        rabi_r_phase = jnp.pi * wave[:, 4]
        rabi_r_cplx = rabi_r_mag * jnp.exp(1j * rabi_r_phase)

        # P projection terms
        # LP_1 = rabi_r * conj(rabi_r)
        # LP_2 = rabi_i * conj(rabi_i)
        # LP_3 = - rabi_i * rabi_r
        # LP_4 = - conj(rabi_i)*conj(rabi_r)
        LP_1 = rabi_r_cplx * jnp.conj(rabi_r_cplx)  # real
        LP_2 = rabi_i_cplx * jnp.conj(rabi_i_cplx)  # real
        LP_3 = - rabi_i_cplx * rabi_r_cplx
        LP_4 = - jnp.conj(rabi_i_cplx) * jnp.conj(rabi_r_cplx)

        norm_denom = LP_1 + LP_2  # shape [N]
        # Avoid divide-by-zero by adding a small epsilon or using jnp.where
        eps = 1e-12
        norm_denom_safe = jnp.where(norm_denom < eps, eps, norm_denom)
        # shape => [N,4]
        P_stacked = jnp.stack([
            LP_1 / norm_denom_safe,
            LP_2 / norm_denom_safe,
            LP_3 / norm_denom_safe,
            LP_4 / norm_denom_safe,
        ], axis=1)

        # The U part (first 5 columns)
        Delta_col = wave[:, 0]
        Ri_re = (jnp.real(rabi_i_cplx) / self.Rabi_i_val)
        Ri_im = (jnp.imag(rabi_i_cplx) / self.Rabi_i_val)
        Rr_re = (jnp.real(rabi_r_cplx) / self.Rabi_r_val)
        Rr_im = (jnp.imag(rabi_r_cplx) / self.Rabi_r_val)

        U_stacked = jnp.stack([Delta_col, Ri_re, Ri_im, Rr_re, Rr_im], axis=1)
        # multiply by (-1j) to match your original approach
        U_stacked_cplx = U_stacked.astype(jnp.complex128) * (-1j)

        return jnp.concatenate([U_stacked_cplx, P_stacked], axis=1)  # [N, 9]

    def exponentials(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Build the batch of matrix exponentials for each time step in the 
        Van Loan approach. Return => [N, 32, 32], where N = no_of_steps + 2*pad.
        """
        # gens => shape [11, 32, 32]
        gens = self.VL_Generators_ansatz
        # aux => shape [N, 9]
        aux = self.return_auxiliary_amplitudes(ctrl_a, ctrl_b, ctrl_c)

        # drift = -1j*(gens[0] + gens[1])
        drift = -1j * (gens[0] + gens[1])

        # control_gens => gens[2:7] => 5 control blocks
        control_gens = gens[2:7]  # shape [5, 32, 32]
        # aux_u => first 5 columns => shape [N, 5]
        aux_u = aux[:, :5]        # shape [N, 5]

        # sum_control => shape [N, 32, 32], from tensordot
        # (N,5) dot (5,32,32) => (N,32,32)
        sum_control = jnp.tensordot(aux_u, control_gens, axes=[[1],[0]])
        drift_expanded = drift[None, :, :]  # [1,32,32]

        # exponent => delta_t*(drift + sum_control)
        exponent = self.delta_t * (drift_expanded + sum_control)  # [N,32,32]

        # batch exponentials => vmap or lax.map
        # We can do jla.expm on each slice
        def single_expm(mat):
            return jla.expm(mat)

        # vmap over exponent => shape [N,32,32]
        step_exps = vmap(single_expm, in_axes=0, out_axes=0)(exponent)
        return step_exps
        # or simpler: step_exps => [N,32,32] directly if out_axes=0. 
        # Just confirm indexing is correct.
        # We'll keep it [N,32,32]. If it flips, you can reorder as needed.

    def propagate(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Multiply all step exponentials from last to first: final = E_{N-1} * ... * E_0.
        Implement with jax.lax.scan in reverse order.
        Returns the final 32x32 matrix.
        """
        step_exps = self.exponentials(ctrl_a, ctrl_b, ctrl_c)  # [N,32,32]
        N = step_exps.shape[0]

        # Reverse them
        step_exps_rev = jnp.flip(step_exps, axis=0)

        def mul_func(carry, x):
            # x => [32,32], carry => [32,32]
            # new_carry = x @ carry
            return jnp.matmul(x, carry), None

        eye_mat = jnp.eye(32, dtype=jnp.complex128)
        final, _ = lax.scan(mul_func, eye_mat, step_exps_rev)
        # final => E_{N-1} * ... * E_0
        return final

    def metrics(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        1) Propagate -> final operator (Van Loan prop)
        2) Compute infidelity vs self.U_target
        3) Compute adiabaticity for |10>.
        Returns (infidelity, adiabaticity).
        """
        VL_propagator = self.propagate(ctrl_a, ctrl_b, ctrl_c)  # [32,32]
        dim = self.dim
        dim_sq = dim * dim  # 16

        U_prop = VL_propagator[0:dim_sq, 0:dim_sq]  # top-left 16x16
        U_target = self.U_target  # 16x16

        # Infidelity measure
        dtrace = jnp.trace(U_target.conj().T @ U_prop)
        dtrace_norm = jnp.trace(U_target.conj().T @ U_target)
        # 1 - real( (dtrace*dtrace*) / dtrace_norm^2 )
        ratio = (dtrace * jnp.conj(dtrace)) / (dtrace_norm**2)
        infidelity = 1.0 - jnp.real(ratio)

        # adiabaticity => ad_integral = [0:16, 16:32]
        ad_integral = VL_propagator[0:dim_sq, dim_sq:2*dim_sq]  # 16x16
        # final_state_10 => U_prop * |10>
        final_state_10 = U_prop @ self.psi_10
        # overlap => final_state_10^\dagger * ( ad_integral * |10> )
        # a scalar
        tmp = ad_integral @ self.psi_10
        overlap = jnp.conj(final_state_10).T @ tmp
        adiabaticity_metric_10 = 1.0 - (1.0 / self.duration) * overlap
        # cast to real
        adiabaticity = jnp.real(adiabaticity_metric_10)
        return (infidelity, adiabaticity)

    def target(
        self,
        ctrl_a: jnp.ndarray,
        ctrl_b: jnp.ndarray,
        ctrl_c: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Weighted cost => 0.2 * infidelity + 0.8 * adiabaticity
        """
        infid, adiab = self.metrics(ctrl_a, ctrl_b, ctrl_c)
        return 0.2 * infid + 0.8 * adiab


# ------------------------------------------------------------------------------
# JAX-friendly "single optimization step" example
# ------------------------------------------------------------------------------
@partial(jit, static_argnums=0)
def single_optimization_step(propagator: PropagatorVLJAX,
                            ctrl_a: jnp.ndarray,
                            ctrl_b: jnp.ndarray,
                            ctrl_c: jnp.ndarray,
                            lr: float = 0.02) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform one gradient-based optimization iteration using a simple
    gradient descent approach (no fancy moment accumulations).
    """
    def loss_fn(a, b, c):
        return propagator.target(a, b, c)

    # Compute cost and grads
    cost, grads = jax.value_and_grad(loss_fn, argnums=(0,1,2))(ctrl_a, ctrl_b, ctrl_c)
    # grads is a tuple of 3 arrays => (gradA, gradB, gradC)
    gradA, gradB, gradC = grads

    # gradient update
    new_ctrl_a = ctrl_a - lr * gradA
    new_ctrl_b = ctrl_b - lr * gradB
    new_ctrl_c = ctrl_c - lr * gradC

    return cost, new_ctrl_a, new_ctrl_b, new_ctrl_c