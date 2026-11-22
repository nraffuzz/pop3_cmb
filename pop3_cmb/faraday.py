import numpy as np
from scipy.integrate import simpson
from .profiles import get_SN_profile_k, get_halo_profile_k
from .starIMF import IMF
from .reionization import Reionization
from typing import Optional
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import brentq

class FaradayModel:
    def __init__(self, camb_runner, params, constants, k_grid, Plin_grid, freq = 1 * 10**9, Model = 'FlatIMF', M_low = 10, M_high = 100,
                 correction = False,
                 nuisance: Optional[dict] = None,
                 gamma: Optional[float] = None, 
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 Mch: Optional[float] = None,
                 sigma: Optional[float] = None,
                 z_re: Optional[float] = None,           
                 Delta_z: Optional[float] = None):

        """_summary_

        Args:
            camb_runner (CAMBRunner): instance of CAMBRunner used to get cosmological information (see cosmology.py)
            params (dict): parameter model for Pop III star formation and SN physics (see config.py)
            constants (dict): constants used for scaling relations and reference values (see config.py)
            k_grid (1D np.ndarray): wavenumber grid
            Plin_grid (2D np.ndarray): linear power spectrum on the grid defined by k_grid and z_grid
            freq (<type>): frequency of observation for generated spectra  

 
            model (string): IMF model
            - PowerIMF : power law IMF with free gamma param
            - FlatIMF : power law IMF with gamma = 0 
            - LogNormalIMF : Log normal IMF with Mch and sigma as free param
            - ChabrierIMF: Chabrier IMF with free alpha and beta param
            - LarsonIMF: Chabrier IMF with alpha = -2.35 and beta = 1

            M_star (1D np.ndarray): amount of stellar mass in halo of a range of masses
            M_sn_max (float): maximum stellar mass for CCSN
            M_sn_min (float): minimum stellar mass for CCSN 
            M_low (float): minimum stellar mass of population 
            M_high (float): maximum stellar mass of population 
            Correction (bool): include parameter correction
            Nuisance (dict): Dictionary specifying 'A' and 'B' values 
            
            Note: Gamma, Sigma, and Mch is defined positive. M_low and M_high must include [10-40]
        """
        self.k = k_grid
        self.Plin = Plin_grid
        self.cr = camb_runner
        self.p = params.copy()
        self.p['nu_Hz'] = freq
        self.c = constants

        self.IMF = Model
        self.M_low = M_low
        self.M_high = M_high
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.Mch = Mch
        self.sigma = sigma

        self.z_re = z_re
        self.Delta_z = Delta_z
        self.nuisance = nuisance if nuisance is not None else {}
        
        self.h = self.cr.h
        self.Om = self.cr.pars.omegam
        self.Ob = self.cr.pars.omegab
        self.z_arr = self.cr.z_grid
        self.nz = len(self.z_arr)
        
        self.M_grid = np.logspace(np.log10(self.p['M_halo_min']), np.log10(self.p['M_halo_max']), self.p['n_M_points'])
        self.rho_m_0 = self.c['rho_crit0_ref'] * self.h**2 * self.Om 

        self.z_time = np.linspace(0, 100, 10001)

        self.t_grid = self.cr.results.physical_time(self.z_time)
        self.t_of_z = PchipInterpolator(self.z_time, self.t_grid)
        self.z_of_t = PchipInterpolator(self.t_grid[::-1], self.z_time[::-1])

        if correction:
            if ('A' not in self.nuisance or 'B' not in self.nuisance or self.z_re is None or self.Delta_z is None):
                raise ValueError(
                    "correction=True requires nuisance['A'], nuisance['B'], "
                    "z_re, and Delta_z."
                    )

            self.reion = Reionization(
                self.cr,
                self.p,
                self.nuisance,
                self.z_re,
                self.Delta_z
            )

        self._init_halo_properties()
        self._init_popIII_properties(correction)

    def active_time(self, z_obs):  # Returns Gyr
        t_obs = self.t_of_z(z_obs)

        # params
        f_b = self.Ob / self.Om
        Delta_c = 18 * np.pi**2
        f_He = self.p['f_He']
        m_p = 1.6726219e-24  # g
        Msun_g = self.c['M_sun_kg'] * 1e3
        Mpc_cm = self.c['pc_to_m'] * self.c['Mpc_to_pc'] * 1e2

        rho_crit0 = self.c['rho_crit0_ref'] * self.h**2
        rho_crit0 *= Msun_g / Mpc_cm**3  # g/cm^3

        E51 = self.p['E_sn'] / 1e51

        def t_rad(z):
            Ez2 = self.Om * (1 + z)**3 + (1 - self.Om)
            rho_crit_z = rho_crit0 * Ez2
            rho_1 = f_b * Delta_c * rho_crit_z
            n_1 = rho_1 / ((1 + 4*f_He) * m_p)
            return 4.9e4 * E51**0.22 * n_1**(-0.55) / 1e9  # Gyr

        # Solve for explosion redshift
        def equation(z_SN):
            return t_obs - self.t_of_z(z_SN) - t_rad(z_SN)

        z_max = self.z_time[-1]
        if equation(z_max) <= 0:
            raise ValueError("No explosion redshift within the time grid.")
        
        z_SN = brentq(equation, z_obs, z_max)
        return t_obs - self.t_of_z(z_SN)
    
    def sedov_time(self, M_avg):  # Returns Gyr
        f_b = self.Ob / self.Om
        Delta_c = 18 * np.pi**2

        Msun_g = self.c['M_sun_kg'] * 1e3
        Mpc_cm = self.c['pc_to_m'] * self.c['Mpc_to_pc'] * 1e2

        M_ej = M_avg * Msun_g
        E_SN = self.p['E_sn']

        rho_crit0 = self.c['rho_crit0_ref'] * self.h**2
        rho_crit0 *= Msun_g / Mpc_cm**3

        v_ej = np.sqrt(2 * E_SN / M_ej)

        def t_ST(z):
            Ez2 = self.Om * (1 + z)**3 + (1 - self.Om)
            rho_1 = f_b * Delta_c * rho_crit0 * Ez2

            R_ST = (3 * M_ej / (4 * np.pi * rho_1))**(1/3)

            return R_ST / v_ej / (365.25 * 24 * 3600 * 1e9)

        t_ST_arr = np.zeros_like(self.z_arr, dtype=float)
        z_max = self.z_time[-1]

        for i, z_obs in enumerate(self.z_arr):
            t_obs = self.t_of_z(z_obs)

            def equation(z_SN):
                return t_obs - self.t_of_z(z_SN) - t_ST(z_SN)

            if equation(z_max) <= 0:
                raise ValueError(f"No valid explosion redshift for z={z_obs}")

            z_SN = brentq(equation, z_obs, z_max)
            t_ST_arr[i] = t_obs - self.t_of_z(z_SN)

        return t_ST_arr
    
    def _M_thr(self, z):
        """Eq. (25): threshold halo mass for T_vir = 1e4 K atomic cooling."""
        Tvir = 1.0e4
        Om_z = self.Om * (1+z)**3 / (self.Om*(1+z)**3 + (1.0 - self.Om))
        d = Om_z - 1.0
        Delta_c = 18*np.pi**2 + 82*d - 39*d**2          # Bryan & Norman fit
        # mu = 0.6
        M_thr = 3.5e7 / self.h * (Tvir/1e4)**1.5 \
            * (self.Om/Om_z * Delta_c/(18*np.pi**2))**(-0.5) \
            * ((1+z)/10.0)**(-1.5)
        return M_thr # M_sun
    
    def tau_star(self, m):
        """
        # Pop III stellar lifetime fit from Schaerer (2002), Table 6, for zero-metallicity stars:
        # log10(tau_*/yr) = 9.785 - 3.759 x + 1.413 x^2 - 0.186 x^3,
        # where x = log10(m/M_sun). Valid for approximately 5-500 M_sun.
        """

        x = np.log10(m)
        log_tau_yr = (
            9.785
            - 3.759*x
            + 1.413*x**2
            - 0.186*x**3
        )
        return 10**log_tau_yr / 1e9  # Gyr
        
    def _init_halo_properties(self):
        """
        Compute halo properties: mass function dndM, bias, and virial radius Rvir on the grid defined by z_arr and M_grid.
        Uses Sheth-Tormen mass function and bias as an example, but can be replaced with other models if desired.
        """
        self.dndM = np.zeros((self.nz, len(self.M_grid)))
        self.bias = np.zeros((self.nz, len(self.M_grid)))
        self.Rvir = np.zeros((self.nz, len(self.M_grid)))
        self.sig2 = np.zeros((self.nz, len(self.M_grid)))  # not needed
        
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
                sig2[j] = simpson(integrand, x=self.k) / (2 * np.pi**2) # Halo mass varience on (M,z)
            self.sig2[i, :] = sig2

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

        P_z0 = self.cr.PK.P(0.0, self.k)
        self.S0 = np.zeros_like(self.M_grid)

        for j, R in enumerate(R_L):
            x = self.k * R
            W = 3 * (np.sin(x) - x*np.cos(x)) / x**3
            integrand = self.k**2 * P_z0 * W**2

            self.S0[j] = simpson(
                integrand,
                x=self.k
            ) / (2 * np.pi**2)

        self.EPS()

    def EPS(self, q = 2.2):
        delta_c = 1.686
        S_interp = PchipInterpolator(np.log(self.M_grid), np.log(self.S0), extrapolate=True) # NOTE: this extrapolates for the lowest mass grid values; can be fixed if needed

        k_ref = 0.05
        ik = np.argmin(np.abs(self.k - k_ref))

        P0 = self.cr.PK.P(0.0, self.k[ik])  
        P_growth = self.cr.PK.P(self.z_time, self.k[ik])
     
        D_grid = np.sqrt(P_growth / P0)
        D_interp = CubicSpline(self.z_time, D_grid) # need derivative thats why we use CubicSpline

        def dMdz(z, M):
            S_M = np.exp(S_interp(np.log(M)))
            S_Mq = np.exp(S_interp(np.log(M / q)))
            D = D_interp(z)

            dD_dz = D_interp(z, 1) # first derivative

            return (np.sqrt(2 / np.pi) * M / np.sqrt(S_Mq - S_M) * delta_c/ D**2 * dD_dz)

        self.EPS_solutions = []

        tau_max = self.tau_star(self.p['M_sn_min'])      # Gyr

        for z_obs in self.z_arr:

            t_obs = self.t_of_z(z_obs)
            t_active_Gyr = self.active_time(z_obs)

            z_needed = self.z_of_t(
                t_obs - tau_max - t_active_Gyr
            )

            sol = solve_ivp(
                dMdz,
                (z_obs, z_needed),
                self.M_grid,
                dense_output=True,
                rtol=1e-8,
                atol=1e-3
            )

            self.EPS_solutions.append(sol)

    def _Mh_active_bracket(self, i, m, t_active_Gyr): # i is redshift index; m is star mass
        """
        Suppression factor for all halo masses in self.M_grid for a given observation redshift and stellar mass.
        """

        tau = self.tau_star(m)

        z_obs = self.z_arr[i]
        t_obs = self.t_of_z(z_obs)
        z_recent = self.z_of_t(t_obs - tau)
        z_early  = self.z_of_t(t_obs - tau - t_active_Gyr)

        sol = self.EPS_solutions[i] # solution for halo grid observed at z_obs

        M_recent = sol.sol(z_recent) # arrays over M grid
        M_early  = sol.sol(z_early)

        return M_recent - M_early

    def _alpha0(self): # Mpc⁻¹
        t_ratio  = self.p['t_age'] / self.c['t_age_ref']      # /1e6 yr
        E_ratio  = self.p['E_sn']  / self.c['E_sn_ref']       # /1e53 erg  -- SEE 4c
        fb_ratio = self.p['fB']    / self.c['fB_ref']
        frel_ratio = self.p['frel'] / self.c['frel_ref']
        nu_ratio = self.p['nu_Hz'] / self.c['nu_Hz_ref']      # /1 GHz
        z_term   = ((1 + self.z_arr) / 20.0)**(3/5)

        alpha0_pc = 2.487001e-15 * t_ratio**(-12/5) * E_ratio**(4/5) * z_term * fb_ratio * frel_ratio * nu_ratio**(-3) # [pc^-1]
        self.alpha0 = alpha0_pc * self.c['Mpc_to_pc']                 # [Mpc^-1], ×1e6
        
    def _init_popIII_properties(self, correction):
        """
        Compute Pop III star formation properties: number of SNe per halo N_sn and Faraday Conversion rate alpha0 on the grid defined by z_arr and M_grid.
        Uses simple scaling relations based on the parameters and constants provided, but can be replaced with more detailed models if desired.
        """
        f_b = self.Ob / self.Om
        M_sn_max, M_sn_min = self.p['M_sn_max'], self.p['M_sn_min']

        self.N_sn = np.zeros((self.nz, len(self.M_grid)))

        for i in range(self.nz):
            t_active_Gyr = self.active_time(self.z_arr[i])
            self.N_sn[i,:] = IMF(self.IMF, lambda m: self._Mh_active_bracket(i, m, t_active_Gyr), self.p['epsilon_star'], f_b, M_sn_max, M_sn_min, self.M_low, self.M_high, self.gamma, self.alpha, self.beta, self.Mch, self.sigma)

        if correction:
            if self.nuisance == {} or self.z_re == None or self.Delta_z == None: 
                raise(ValueError)
            cor = self.reion.N_SN_cor_factor()
            self.N_sn = self.N_sn * cor[:, None]

        self._alpha0()
    
    def compute_P_alpha(self, M_avg = 21.64, return_components=False):
        """
        Compute Faraday Conv. power spectrum P_alpha(k, z) using the halo model.
        Returns P1h + P2h on the (z_arr, k) grid.
        """
        P1h = np.zeros((self.nz, len(self.k)))
        P2h = np.zeros((self.nz, len(self.k)))
        dM  = np.gradient(np.log(self.M_grid))

        t_ST = self.sedov_time(M_avg) * 1e9
        t_rad = np.array([self.active_time(z) for z in self.z_arr]) * 1e9

        for i in range(self.nz):
            z = self.z_arr[i]
            t_min = t_ST[i]
            t_max = t_rad[i]

            t_ages = np.geomspace(t_min, t_max, 128)
            alpha_samples = []
            for t_age in t_ages:
                rs_phys_pc = 256.261223249 * (self.p['E_sn']/self.c['E_sn_ref'])**(1/5) \
                    * (self.c['ombh2_ref']/0.0245)**(-1/5) \
                    * ((1 + z)/20.0)**(-3/5) \
                    * (t_age/self.c['t_age_ref'])**(2/5)

                rs_phys_Mpc = rs_phys_pc * 1e-6
                rp_phys_Mpc = rs_phys_Mpc * (self.p['eta']/(self.p['eta']-1))**(-1/3)
                V_rem_phys  = (4*np.pi/3) * (rs_phys_Mpc**3 - rp_phys_Mpc**3)   # proper Mpc^3

                u_sn = get_SN_profile_k(self.k * (1 + z), rs_phys_Mpc, self.p['eta'])  # dimensionless (converted k back to physical)
                alpha0_t = self.alpha0[i] * (t_age / self.p['t_age'])**(-12/5)
                alpha_tilde_phys = alpha0_t * V_rem_phys * u_sn
                alpha_samples.append(alpha_tilde_phys)

            alpha_samples = np.asarray(alpha_samples)
            alpha_tilde_avg = np.trapezoid(
                alpha_samples,
                x=t_ages,
                axis=0
            ) / (t_max - t_min)

            alpha_tilde_sq = np.trapezoid(
                alpha_samples**2,
                x=t_ages,
                axis=0
            ) / (t_max - t_min)

            alpha_tilde_sq *= (1 + z)**4
            alpha_tilde = alpha_tilde_avg * (1 + z)**2          # -> comoving Mpc^2

            M_thr = self._M_thr(z)
            mask  = self.M_grid >= M_thr
            
            bracket_self = np.zeros_like(self.k) # Mpc⁻³
            bracket_pairs = np.zeros_like(self.k) # Mpc⁻³
            bracket_2 = np.zeros_like(self.k) # Mpc⁻³

            for j in range(len(self.M_grid)):
                if not mask[j]:                   # skip sub-threshold halos
                    continue
                N = self.N_sn[i,j]
                u_halo = get_halo_profile_k(self.k, self.Rvir[i,j], 10.0)

                bracket_self += self.dndM[i,j] * N * dM[j]
                bracket_pairs += self.dndM[i,j] * N**2 * u_halo**2 * dM[j] # see overleaf Poisson distrabution around N (mean)
                bracket_2 += self.dndM[i,j] * self.bias[i,j] * N * dM[j]

            P1h[i, :] = alpha_tilde_sq * bracket_self + alpha_tilde**2 * bracket_pairs # Mpc⁴ · Mpc⁻³ = Mpc
            P2h[i, :] = alpha_tilde**2 * self.Plin[i, :] * bracket_2**2 # alpha_tilde² · Plin · bracket_2² = Mpc⁴ · Mpc³ · Mpc⁻⁶ = Mpc

        if return_components:
            return P1h, P2h
        else:
            return P1h + P2h