import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from jax.numpy import trapezoid
from functools import partial
from .profiles_JAX import get_SN_profile_k, get_halo_profile_k


# outsource the math with JIT helper functions
@jax.jit
def _M_thr_JIT(z, Om, h):
    Tvir = 1.0e4
    Om_z = Om * (1+z)**3 / (Om*(1+z)**3 + (1.0 - Om))
    d = Om_z - 1.0
    Delta_c = 18*jnp.pi**2 + 82*d - 39*d**2          # Bryan & Norman fit
    # mu = 0.6
    M_thr = 3.5e7 / h * (Tvir/1e4)**1.5 \
        * (Om/Om_z * Delta_c/(18*jnp.pi**2))**(-0.5) \
        * ((1+z)/10.0)**(-1.5)
    return M_thr # M_sun

@jax.jit
def init_halo_properties_JIT(k, Plin, z_arr, M_grid, h, Om, rho_crit0_ref, A, a, p):

    rho_m_0 = rho_crit0_ref * h**2 * Om
    delta_c = 1.686
    R_L = (3 * M_grid / (4 * jnp.pi * rho_m_0))**(1/3) # Lagrangian radius
    # R_L is defined at the mean matter density today and is comoving,
    # exactly what the top-hat window over the comoving P(k) needs
    
    rho_crit0 = rho_crit0_ref * h**2  # M_sun/Mpc^3 (physical, z=0)
    OmL = 1.0 - Om                              # flat LCDM
    Delta_c = 18 * jnp.pi**2                          # ~178
    
    # z_arr (,nZ)
    # Plin (nz,nK)
    sig2 = jnp.zeros_like(M_grid) # (,nM)

    x = R_L[:, None] * k[None, :] # (nM, nK)
    W = 3 * (jnp.sin(x) - x*jnp.cos(x)) / (x**3) # (nM, nK)
    integrand = k[None, None, :]**2 * Plin[None, :, :] * W[:, None, :]**2 # (nM, nZ, nK)
    sig2 = trapezoid(integrand, x=k, axis = 2) / (2 * jnp.pi**2) # (nM, nZ)

    sigma = jnp.sqrt(sig2) # (nM, nZ)
    nu = (delta_c / sigma)**2 # (nM, nZ)
    nu_prime = a * nu # (nM, nZ)
    f_nu = A * jnp.sqrt(2*nu_prime/jnp.pi) * (1 + nu_prime**(-p)) *jnp.exp(-nu_prime/2) # (nM, nZ)

    dlnsigma_dlnM = jnp.gradient(jnp.log(sigma), jnp.log(M_grid), axis = 0) # axis does gradient along mass 
    dndM = (rho_m_0 / M_grid[:, None]) * f_nu * jnp.abs(dlnsigma_dlnM) # dn/dlnM (units Mpc⁻³)
    bias = 1 + (nu_prime - 1)/delta_c + (2*p/delta_c)/(1 + nu_prime**p)

    E2 = Om * (1 + z_arr)**3 + OmL # (,nZ)
    rho_crit = rho_crit0 * E2                # physical M_sun/Mpc^3 
    Rvir_phys = (3 * M_grid[:, None] / (4 * jnp.pi * Delta_c * rho_crit[None, :]))**(1/3) # (nM, nZ)
    Rvir = Rvir_phys * (1 + z_arr)[None, :]                # convert to comoving

    return dndM.T, bias.T, Rvir.T # A: Accidently broadcasted backwards 

@jax.jit
def alpha0_JIT(
    z_arr,
    t_age,
    t_age_ref,
    E_sn,
    E_sn_ref,
    fB,
    fB_ref,
    frel,
    frel_ref,
    nu_Hz,
    nu_Hz_ref,
    Mpc_to_pc,
):
    t_ratio = t_age / t_age_ref
    E_ratio = E_sn / E_sn_ref
    fb_ratio = fB / fB_ref
    frel_ratio = frel / frel_ref
    nu_ratio = nu_Hz / nu_Hz_ref

    z_term = ((1 + z_arr) / 20.0)**(3 / 5)

    alpha0_pc = (
        20.0
        * t_ratio**(-12 / 5)
        * E_ratio**(4 / 5)
        * z_term
        * fb_ratio
        * frel_ratio
        * nu_ratio**(-3)
    )

    return alpha0_pc * Mpc_to_pc

@partial(jax.jit, static_argnames=["nz"])
def _init_popIII_properties_JIT(Ob, Om, epsilon_star, M_grid, M_sn_max, M_sn_min, nz):
        f_b = Ob / Om
        M_star = epsilon_star * f_b * M_grid
        
        M_max, M_min = 100.0, 10.0
        A_imf = M_star / (M_max - M_min)
        
        N_sn_val = A_imf * jnp.log(M_sn_max / M_sn_min)
        
        N_sn = jnp.tile(N_sn_val, (nz, 1))
        return N_sn

@jax.jit
def compute_P_alpha_JIT(E_sn, E_sn_ref, ombh2_ref, z_arr, t_age, t_age_ref, eta, k, alpha0, M_grid, Rvir, dndM ,N_sn, bias, Plin, Om, h):

    dM  = jnp.gradient(jnp.log(M_grid))


    rs_phys_pc = 2.0 * (E_sn/E_sn_ref)**(1/5) \
        * (ombh2_ref/0.0245)**(-1/5) \
        * ((1 + z_arr)/20.0)**(-3/5) \
        * (t_age/t_age_ref)**(2/5) # (,nZ)

    rs_phys_Mpc = rs_phys_pc * 1e-6 # (,nZ)
    rp_phys_Mpc = rs_phys_Mpc * (eta/(eta-1))**(-1/3) # (,nZ)
    V_rem_phys  = (4*jnp.pi/3) * (rs_phys_Mpc**3 - rp_phys_Mpc**3)   # proper Mpc^3 # (,nZ)

    u_sn = get_SN_profile_k(k, rs_phys_Mpc, eta)  # dimensionless # (nZ, nK)
    alpha_tilde_phys = alpha0[:, None] * V_rem_phys[:, None] * u_sn # proper Mpc^2 (nZ, nK)

    alpha_tilde = alpha_tilde_phys * (1 + z_arr)[:, None]**2          # -> comoving Mpc^2

    M_thr = _M_thr_JIT(z_arr, Om, h) # (,nZ)
    mask  = M_grid[None, :] >= M_thr[:, None] # (nZ, nM)

    active = mask.astype(k.dtype)[:, :, None] # (1, nM, 1)
    u_halo = get_halo_profile_k(k, Rvir, 10.0) # (nZ, nM, nK)
    bracket_1 = jnp.sum(active * dndM[:, :, None] * ( N_sn[:, :, None] + N_sn[:, :, None]*(N_sn[:, :, None]-1)*u_halo**2 ) * dM[None, :, None], axis = 1) # sum over masses
    bracket_2 = jnp.sum(active * dndM[:, :, None] * bias[:, :, None] * N_sn[:, :, None] * dM[None, :, None], axis = 1)

    P1h = alpha_tilde**2 * bracket_1
    P2h = alpha_tilde**2 * Plin * bracket_2**2

    return P1h, P2h


class FaradayModel:
    def __init__(self, camb_runner, params, constants, k_grid, Plin_grid):
        """_summary_

        Args:
            camb_runner (CAMBRunner): instance of CAMBRunner used to get cosmological information (see cosmology.py)
            params (dict): parameter model for Pop III star formation and SN physics (see config.py)
            constants (dict): constants used for scaling relations and reference values (see config.py)
            k_grid (1D np.ndarray): wavenumber grid
            Plin_grid (2D np.ndarray): linear power spectrum on the grid defined by k_grid and z_grid
        """
        self.k = k_grid
        self.Plin = Plin_grid
        self.cr = camb_runner
        self.p = params
        self.c = constants
        
        self.h = self.cr.h
        self.Om = self.cr.pars.omegam
        self.Ob = self.cr.pars.omegab
        self.z_arr = jnp.array(self.cr.z_grid)
        self.nz = len(self.z_arr)
        
        self.M_grid = jnp.logspace(jnp.log10(self.p['M_halo_min']), jnp.log10(self.p['M_halo_max']), self.p['n_M_points'])
        self.rho_m_0 = self.c['rho_crit0_ref'] * self.h**2 * self.Om 
        
        self._init_halo_properties()
        self._init_popIII_properties()
    
    def _M_thr(self, z):
        """Eq. (25): threshold halo mass for T_vir = 1e4 K atomic cooling."""
        return _M_thr_JIT(z, self.Om, self.h)
        
    def _init_halo_properties(self):
        """
        Compute halo properties: mass function dndM, bias, and virial radius Rvir on the grid defined by z_arr and M_grid.
        Uses Sheth-Tormen mass function and bias as an example, but can be replaced with other models if desired.
        """
        self.dndM, self.bias, self.Rvir = init_halo_properties_JIT(self.k, self.Plin, self.z_arr, self.M_grid, self.h, self.Om, self.c["rho_crit0_ref"], self.p["st_A"], self.p["st_a"], self.p["st_p"])
        
    def _alpha0(self):
        self.alpha0 = alpha0_JIT(
            self.z_arr,
            self.p["t_age"],
            self.c["t_age_ref"],
            self.p["E_sn"],
            self.c["E_sn_ref"],
            self.p["fB"],
            self.c["fB_ref"],
            self.p["frel"],
            self.c["frel_ref"],
            self.p["nu_Hz"],
            self.c["nu_Hz_ref"],
            self.c["Mpc_to_pc"],
        )
            
    def _init_popIII_properties(self):
        """
        Compute Pop III star formation properties: number of SNe per halo N_sn and Faraday Conversion rate alpha0 on the grid defined by z_arr and M_grid.
        Uses simple scaling relations based on the parameters and constants provided, but can be replaced with more detailed models if desired.
        """

        self.N_sn = _init_popIII_properties_JIT(self.Ob, self.Om, self.p['epsilon_star'], self.M_grid, self.p['M_sn_max'], self.p['M_sn_min'], self.nz)
        self._alpha0()
    
    def compute_P_alpha(self, return_components=False):
        """
        Compute Faraday Conv. power spectrum P_alpha(k, z) using the halo model.
        Returns P1h + P2h on the (z_arr, k) grid.
        """
        P1h, P2h = compute_P_alpha_JIT(self.p['E_sn'], self.c['E_sn_ref'], self.c['ombh2_ref'], self.z_arr, self.p['t_age'], self.c['t_age_ref'], self.p['eta'], self.k, self.alpha0, self.M_grid, self.Rvir, self.dndM, self.N_sn, self.bias, self.Plin, self.Om, self.h)

        if return_components == True: 
            return P1h, P2h
        else:
            return P1h + P2h
