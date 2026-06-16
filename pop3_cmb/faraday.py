import numpy as np
from scipy.integrate import simpson
from .profiles import get_SN_profile_k, get_halo_profile_k

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
        self.z_arr = self.cr.z_grid
        self.nz = len(self.z_arr)
        
        self.M_grid = np.logspace(np.log10(self.p['M_halo_min']), np.log10(self.p['M_halo_max']), self.p['n_M_points'])
        self.rho_m_0 = self.c['rho_crit0_ref'] * self.h**2 * self.Om 
        
        self._init_halo_properties()
        self._init_popIII_properties()
    
    def _M_thr(self, z):
        """Eq. (25): threshold halo mass for T_vir = 1e4 K atomic cooling."""
        Tvir = 1.0e4
        Om_z = self.Om * (1+z)**3 / (self.Om*(1+z)**3 + (1.0 - self.Om))
        d = Om_z - 1.0
        Delta_c = 18*np.pi**2 + 82*d - 39*d**2          # Bryan & Norman fit
        # mu = 0.6
        M_thr = 3.5e7 / self.h * (Tvir/1e4)**1.5 \
            * (self.Om/Om_z * Delta_c/(18*np.pi**2))**(-1) \
            * ((1+z)/10.0)**(-1.5)
        return M_thr # M_sun
        
    def _init_halo_properties(self):
        """
        Compute halo properties: mass function dndM, bias, and virial radius Rvir on the grid defined by z_arr and M_grid.
        Uses Sheth-Tormen mass function and bias as an example, but can be replaced with other models if desired.
        """
        self.dndM = np.zeros((self.nz, len(self.M_grid)))
        self.bias = np.zeros((self.nz, len(self.M_grid)))
        self.Rvir = np.zeros((self.nz, len(self.M_grid)))
        
        delta_c = 1.686
        R_L = (3 * self.M_grid / (4 * np.pi * self.rho_m_0))**(1/3) # Lagrangian radius
        # R_L is defined at the mean matter density today and is comoving,
        # exactly what the top-hat window over the comoving P(k) needs
        
        rho_crit0 = self.c['rho_crit0_ref'] * self.h**2  # M_sun/Mpc^3 (physical, z=0)
        OmL = 1.0 - self.Om                              # flat LCDM
        Delta_c = 18 * np.pi**2                          # ~178
        
        for i, z in enumerate(self.z_arr):
            P_z = self.Plin[i, :]
            sig2 = np.zeros_like(self.M_grid)
            for j, R in enumerate(R_L):
                x = self.k * R
                W = 3 * (np.sin(x) - x*np.cos(x)) / (x**3)
                integrand = self.k**2 * P_z * W**2
                sig2[j] = simpson(integrand, x=self.k) / (2 * np.pi**2)
            
            sigma_z = np.sqrt(sig2)
            nu = (delta_c / sigma_z)**2
            A, a, p = self.p['st_A'], self.p['st_a'], self.p['st_p']
            nu_prime = a * nu
            f_nu = A * np.sqrt(2*nu_prime/np.pi) * (1 + nu_prime**(-p)) * np.exp(-nu_prime/2)
            
            dlnsigma_dlnM = np.gradient(np.log(sigma_z), np.log(self.M_grid))
            self.dndM[i, :] = (self.rho_m_0 / self.M_grid) * f_nu * np.abs(dlnsigma_dlnM) # dn/dlnM (units Mpc⁻³)
            self.bias[i, :] = 1 + (nu_prime - 1)/delta_c + (2*p/delta_c)/(1 + nu_prime**p)
            # self.Rvir[i, :] = R_L
            
            Ez2 = self.Om * (1 + z)**3 + OmL
            rho_crit_z = rho_crit0 * Ez2                          # physical M_sun/Mpc^3 at z
            Rvir_phys = (3 * self.M_grid / (4 * np.pi * Delta_c * rho_crit_z))**(1/3)
            self.Rvir[i, :] = Rvir_phys * (1 + z)                 # convert to comoving

        
    def _alpha0(self): # Mpc⁻¹
        t_ratio  = self.p['t_age'] / self.c['t_age_ref']      # /1e6 yr
        E_ratio  = self.p['E_sn']  / self.c['E_sn_ref']       # /1e53 erg  -- SEE 4c
        fb_ratio = self.p['fB']    / self.c['fB_ref']
        frel_ratio = self.p['frel'] / self.c['frel_ref']
        nu_ratio = self.p['nu_Hz'] / self.c['nu_Hz_ref']      # /1 GHz
        z_term   = ((1 + self.z_arr) / 20.0)**(3/5)

        alpha0_pc = 20.0 * t_ratio**(-12/5) * E_ratio**(4/5) * z_term * fb_ratio * frel_ratio * nu_ratio**(-3) # [pc^-1]
        self.alpha0 = alpha0_pc * self.c['Mpc_to_pc']                 # [Mpc^-1], ×1e6
        

    def _init_popIII_properties(self):
        """
        Compute Pop III star formation properties: number of SNe per halo N_sn and Faraday Conversion rate alpha0 on the grid defined by z_arr and M_grid.
        Uses simple scaling relations based on the parameters and constants provided, but can be replaced with more detailed models if desired.
        """
        f_b = self.Ob / self.Om
        M_star = self.p['epsilon_star'] * f_b * self.M_grid
        
        M_max, M_min = 100.0, 10.0
        A_imf = M_star / (M_max - M_min)
        
        M_sn_max, M_sn_min = self.p['M_sn_max'], self.p['M_sn_min']
        N_sn_val = A_imf * np.log(M_sn_max / M_sn_min)
        
        self.N_sn = np.tile(N_sn_val, (self.nz, 1))
        self._alpha0()
    
    def compute_P_alpha(self, return_components=False):
        """
        Compute Faraday Conv. power spectrum P_alpha(k, z) using the halo model.
        Returns P1h + P2h on the (z_arr, k) grid.
        """
        P1h = np.zeros((self.nz, len(self.k)))
        P2h = np.zeros((self.nz, len(self.k)))
        dM  = np.gradient(np.log(self.M_grid))
        

        for i in range(self.nz):
            rs_phys_pc = 2.0 * (self.p['E_sn']/self.c['E_sn_ref'])**(1/5) \
                * (self.c['ombh2_ref']/0.0245)**(-1/5) \
                * ((1 + self.z_arr[i])/20.0)**(-3/5) \
                * (self.p['t_age']/self.c['t_age_ref'])**(2/5)

            rs_phys_Mpc = rs_phys_pc * 1e-6
            rp_phys_Mpc = rs_phys_Mpc * (self.p['eta']/(self.p['eta']-1))**(-1/3)
            V_rem_phys  = (4*np.pi/3) * (rs_phys_Mpc**3 - rp_phys_Mpc**3)   # proper Mpc^3

            u_sn = get_SN_profile_k(self.k, rs_phys_Mpc, self.p['eta'])  # dimensionless
            alpha_tilde_phys = self.alpha0[i] * V_rem_phys * u_sn / (2*np.pi)**3  # proper Mpc^2
            z = self.z_arr[i]
            alpha_tilde = alpha_tilde_phys * (1 + z)**2          # -> comoving Mpc^2

            M_thr = self._M_thr(self.z_arr[i])
            mask  = self.M_grid >= M_thr
            
            bracket_1 = np.zeros_like(self.k) # Mpc⁻³
            bracket_2 = np.zeros_like(self.k) # Mpc⁻³
            
            for j in range(len(self.M_grid)):
                if not mask[j]:                   # skip sub-threshold halos
                    continue
                u_halo = get_halo_profile_k(self.k, self.Rvir[i,j], 10.0)
                bracket_1 += self.dndM[i,j] * (
                    self.N_sn[i,j] + self.N_sn[i,j]*(self.N_sn[i,j]-1)*u_halo**2
                    ) * dM[j]
                bracket_2 += self.dndM[i,j] * self.bias[i,j] * self.N_sn[i,j] * dM[j]

            P1h[i, :] = alpha_tilde**2 * bracket_1 # alpha_tilde² · bracket_1 = Mpc⁴ · Mpc⁻³ = Mpc
            P2h[i, :] = alpha_tilde**2 * self.Plin[i, :] * bracket_2**2 # alpha_tilde² · Plin · bracket_2² = Mpc⁴ · Mpc³ · Mpc⁻⁶ = Mpc

        if return_components:
            return P1h, P2h
        else:
            return P1h + P2h