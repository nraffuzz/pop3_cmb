import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from jax.numpy import trapezoid
from functools import partial
from .profiles import get_SN_profile_k, get_halo_profile_k


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
    nz = len(z_arr)
    nM = len(M_grid)

    rho_m_0 = rho_crit0_ref * h**2 * Om

    dndM = jnp.zeros((nz, len(M_grid)))
    bias = jnp.zeros((nz, len(M_grid)))
    Rvir = jnp.zeros((nz, len(M_grid)))
    
    delta_c = 1.686
    R_L = (3 * M_grid / (4 * jnp.pi * rho_m_0))**(1/3) # Lagrangian radius
    # R_L is defined at the mean matter density today and is comoving,
    # exactly what the top-hat window over the comoving P(k) needs
    
    rho_crit0 = rho_crit0_ref * h**2  # M_sun/Mpc^3 (physical, z=0)
    OmL = 1.0 - Om                              # flat LCDM
    Delta_c = 18 * jnp.pi**2                          # ~178
    
    for i in range(nz):
        z = z_arr[i]

        P_z = Plin[i, :]
        sig2 = jnp.zeros_like(M_grid)
        for j in range(nM):
            R = R_L[j]

            x = k * R
            W = 3 * (jnp.sin(x) - x*jnp.cos(x)) / (x**3)
            integrand = k**2 * P_z * W**2
            sig2 = sig2.at[j].set(trapezoid(integrand, x=k) / (2 * jnp.pi**2))
        
        sigma_z = jnp.sqrt(sig2)
        nu = (delta_c / sigma_z)**2
        nu_prime = a * nu
        f_nu = A * jnp.sqrt(2*nu_prime/jnp.pi) * (1 + nu_prime**(-p)) *jnp.exp(-nu_prime/2)
        
        dlnsigma_dlnM = jnp.gradient(jnp.log(sigma_z), jnp.log(M_grid))
        dndM = dndM.at[i, :].set((rho_m_0 / M_grid) * f_nu * jnp.abs(dlnsigma_dlnM)) # dn/dlnM (units Mpc⁻³)
        bias = bias.at[i, :].set(1 + (nu_prime - 1)/delta_c + (2*p/delta_c)/(1 + nu_prime**p))
        # self.Rvir[i, :] = R_L
        
        Ez2 = Om * (1 + z)**3 + OmL
        rho_crit_z = rho_crit0 * Ez2                          # physical M_sun/Mpc^3 at z
        Rvir_phys = (3 * M_grid / (4 * jnp.pi * Delta_c * rho_crit_z))**(1/3)
        Rvir = Rvir.at[i, :].set(Rvir_phys * (1 + z))                 # convert to comoving
    return dndM, bias, Rvir

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

# @partial(jax.jit, static_argnames=["nz"])
def compute_P_alpha_JIT(nz, E_sn, E_sn_ref, ombh2_ref, z_arr, t_age, t_age_ref, eta, k, alpha0, M_grid, Rvir, dndM ,N_sn, bias, Plin, Om, h):

    P1h = jnp.zeros((nz, len(k)))
    P2h = jnp.zeros((nz, len(k)))
    dM  = jnp.gradient(jnp.log(M_grid))

    for i in range(nz):

        z = z_arr[i]
        rs_phys_pc = 2.0 * (E_sn/E_sn_ref)**(1/5) \
            * (ombh2_ref/0.0245)**(-1/5) \
            * ((1 + z)/20.0)**(-3/5) \
            * (t_age/t_age_ref)**(2/5)

        rs_phys_Mpc = rs_phys_pc * 1e-6
        rp_phys_Mpc = rs_phys_Mpc * (eta/(eta-1))**(-1/3)
        V_rem_phys  = (4*jnp.pi/3) * (rs_phys_Mpc**3 - rp_phys_Mpc**3)   # proper Mpc^3

        u_sn = get_SN_profile_k(k, rs_phys_Mpc, eta)  # dimensionless
        alpha_tilde_phys = alpha0[i] * V_rem_phys * u_sn # proper Mpc^2
        
        alpha_tilde = alpha_tilde_phys * (1 + z)**2          # -> comoving Mpc^2

        M_thr = _M_thr_JIT(z, Om, h)
        mask  = M_grid >= M_thr
        
        bracket_1 = jnp.zeros_like(k) # Mpc⁻³
        bracket_2 = jnp.zeros_like(k) # Mpc⁻³
        
        for j in range(len(M_grid)):
            active = mask[j].astype(k.dtype)                  # skip sub-threshold halos

            u_halo = get_halo_profile_k(k, Rvir[i,j], 10.0)
            bracket_1 = bracket_1 + active * dndM[i,j] * (
                N_sn[i,j] + N_sn[i,j]*(N_sn[i,j]-1)*u_halo**2
                ) * dM[j]
            bracket_2 = bracket_2 + active * dndM[i,j] * bias[i,j] * N_sn[i,j] * dM[j]

        P1h = P1h.at[i, :].set(alpha_tilde**2 * bracket_1) # alpha_tilde² · bracket_1 = Mpc⁴ · Mpc⁻³ = Mpc
        P2h = P2h.at[i, :].set(alpha_tilde**2 * Plin[i, :] * bracket_2**2) # alpha_tilde² · Plin · bracket_2² = Mpc⁴ · Mpc³ · Mpc⁻⁶ = Mpc

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
        P1h, P2h = compute_P_alpha_JIT(self.nz, self.p['E_sn'], self.c['E_sn_ref'], self.c['ombh2_ref'], self.z_arr, self.p['t_age'], self.c['t_age_ref'], self.p['eta'], self.k, self.alpha0, self.M_grid, self.Rvir, self.dndM, self.N_sn, self.bias, self.Plin, self.Om, self.h)

        if return_components == True: 
            return P1h, P2h
        else:
            return P1h + P2h
