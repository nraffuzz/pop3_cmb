import numpy as np
import camb
from camb import model

class CAMBRunner:
    def __init__(self, cosmo_params, k_min, k_max, n_k_points, z_min, z_max):
        """Initialization of CAMBRunner with cosmological parameters and grids. 

        Args:
            cosmo_params (dict): set of cosmological parameters (H0, ombh2, omch2, omk, tau, As, ns, r)
            k_min (float): minimum wavenumber
            k_max (float): maximum wavenumber
            n_k_points (int): number of wavenumber points
            z_min (float): minimum redshift
            z_max (float): maximum redshift
        """
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(
            H0=cosmo_params['H0'],
            ombh2=cosmo_params['ombh2'],
            omch2=cosmo_params['omch2'],
            omk=cosmo_params['omk'],
            tau=cosmo_params['tau']
        )
        self.pars.InitPower.set_params(
            As=cosmo_params['As'],
            ns=cosmo_params['ns'],
            r=cosmo_params['r']
        )
        self.z_grid = np.arange(z_min, z_max + 1, 1)
        self.k_min = k_min
        self.k_max = k_max
        self.n_k = n_k_points
        
        # Set matter power calculation
        self.pars.set_matter_power(redshifts=list(self.z_grid), kmax=self.k_max)
        self.h = self.pars.h

    def get_linear_matter_power(self):
        """Returns z_grid, k_grid [1/Mpc], and P_lin [Mpc^3]."""
        self.pars.NonLinear = model.NonLinear_none
        results = camb.get_results(self.pars)
        
        # CAMB uses k/h; convert to 1/Mpc.
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=self.k_min / self.h, 
            maxkh=self.k_max / self.h, 
            npoints=self.n_k
        )
        k_Mpc = kh * self.h
        return np.array(z), k_Mpc, pk

    def get_comoving_distance(self, z_array):
        results = camb.get_results(self.pars)
        return results.comoving_radial_distance(z_array)

    def get_primordial_Cl_EE(self, lmax=2000):
        """Computes primordial C_l^EE.
        Rather than computing the (radially) comoving E-mode power spectrum (later to be converter through limber approximation), 
        we directly compute the primordial C_l^EE from CAMB.
        """
        self.pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        results = camb.get_results(self.pars)
        powers = results.get_cmb_power_spectra(self.pars, CMB_unit='muK', lmax=lmax)
        
        Dl_EE = powers['total'][:, 1]
        ls = np.arange(Dl_EE.shape[0])
        Cl_EE = np.zeros_like(Dl_EE)
        mask = ls > 1
        Cl_EE[mask] = Dl_EE[mask] * 2 * np.pi / (ls[mask] * (ls[mask] + 1))
        
        return ls, Cl_EE