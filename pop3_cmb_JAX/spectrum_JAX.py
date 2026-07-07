from .wignerCache_JAX import WignerKernelCache
import jax
import jax.numpy as jnp
from jax.numpy import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator
jax.config.update("jax_enable_x64", True)


@jax.jit
def compute_Cl_alpha_limber_JIT(l_list, r, k, z, P_alpha):
        valid = r > 0
        r_val = jnp.where(valid, r, 1.0) # (nr,)
        
        k_req = (l_list[:, None] + 0.5) / r_val[None, :] # (nl, nr)

        mask = (valid[None, :] & (k_req >= k[0]) & (k_req <= k[-1]))

        safe_z = jnp.broadcast_to(z[None, :],k_req.shape)
        safe_z = jnp.where(mask, safe_z, z[0])

        safe_k = jnp.where(mask, k_req, k[0])
        P_alpha = RegularGridInterpolator((jnp.asarray(z), jnp.log(k)), jnp.log(P_alpha + 1e-30),method="linear",bounds_error=False,fill_value=-jnp.inf,)

        points = jnp.stack([safe_z, jnp.log(safe_k)], axis=-1)  # (nl, nr, 2)
        log_p = P_alpha(points)
        P_vals = jnp.where(mask, jnp.exp(log_p), 0.0)  # (nl, nr)

        integrand = P_vals / r_val[None, :]**2  # (nl, nr)
        Cl = jnp.trapezoid(integrand, x=r, axis=1)  # (nl,)

        return Cl

@jax.jit
def compute_Cl_VV_JIT(K, Cl_alpha, Cl_EE_cut):
    Cl_VV = Cl_alpha @ K @ Cl_EE_cut

    return Cl_VV

class VVSpectrum:
    def __init__(self, ell_min, ell_max, P_alpha, z_arr, r_arr, k_arr, Cl_EE):
        """
        Initialize the VVSpectrum object.

        Args:
            ell_min (float): min ell for output Cl_VV
            ell_max (float): max ell for output Cl_VV
            P_alpha (2D np.ndarray): Faraday Conversion power spectrum P_alpha(k, z) on the grid defined by z_arr and k_arr
            z_arr (1D np.ndarray): redshift grid
            r_arr (1D np.ndarray): radial grid
            k_arr (1D np.ndarray): wavenumber grid
            Cl_EE (1D np.ndarray): EE power spectrum on the same ell grid as Cl_VV output
        """
        self.ells = jnp.arange(ell_min, ell_max + 1)
        self.P_alpha = P_alpha
        self.z = z_arr
        self.r = r_arr
        self.k = k_arr
        self.Cl_EE = Cl_EE

        self.ell_min = int(ell_min)
        self.ell_max = int(self.ells[-1])

    def compute_Cl_alpha_limber(self, l_list):
        """
        Apply Limber approximation to compute Cl_alpha from P_alpha(k, z). 
        Returns angular power spectrum Cl_alpha on the input list of ell values.
        """        
        return compute_Cl_alpha_limber_JIT(l_list, self.r, self.k, self.z, self.P_alpha)

    def compute_Cl_VV(self):
        """
        Compute Cl_VV by convolving Cl_alpha and Cl_EE with Wigner 3j symbols:
        Equation 11 of https://arxiv.org/abs/1401.1371

        Returns:
            output_ells, Cl_VV
        """
        output_ells = jnp.unique(jnp.logspace(jnp.log10(self.ell_min), jnp.log10(self.ell_max), 25).astype(int))
        l_sum = jnp.arange(2, self.ell_max) # A: +1? 


        Cl_EE_full = jnp.zeros((self.ell_max,), dtype=self.Cl_EE.dtype)

        n = min(self.Cl_EE.shape[0], self.ell_max)
        Cl_EE_full = Cl_EE_full.at[:n].set(self.Cl_EE[:n])
        Cl_EE_cut = Cl_EE_full[l_sum]


        Cl_alpha = self.compute_Cl_alpha_limber(l_sum)
        cache = WignerKernelCache('pop3_cmb_JAX/wigner_cache_ellmax3000/')
        K = cache.get_kernel_for_L_index(output_ells)
        
        return output_ells, compute_Cl_VV_JIT(K, Cl_alpha, Cl_EE_cut)