import numpy as np
from pop3_cmb.config import PARAMS, CONSTANTS, COSMOLOGY
from pop3_cmb.cosmology import CAMBRunner
from pop3_cmb.faraday import FaradayModel
from pop3_cmb.spectrum import VVSpectrum
import matplotlib.pyplot as plt
from time import time

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    camb_start = time()
    camb_run = CAMBRunner(COSMOLOGY, PARAMS['k_min'], PARAMS['k_max'], PARAMS['n_k_points'], PARAMS['z_min'], PARAMS['z_max'])
    z_grid, k_grid, Plin = camb_run.get_linear_matter_power()
    r_grid = camb_run.get_comoving_distance(z_grid)
    ls_ee, Cl_EE = camb_run.get_primordial_Cl_EE(lmax=3000)
    camb_end = time()
    
    fd_start = time()
    # test FlatIMF
    fm = FaradayModel(camb_run, PARAMS, CONSTANTS, k_grid, Plin, 
                    freq = 27 * 10 ** 9,
                    Model = 'FlatIMF',
                    M_low = 10, 
                    M_high= 100)
    
    P_alpha = fm.compute_P_alpha()
    fd_end = time()
    
    spectrum_start = time()
    convolution = VVSpectrum(2, PARAMS['ell_max'], P_alpha, z_grid, r_grid, k_grid, Cl_EE)
    ls, Cl_VV = convolution.compute_Cl_VV()
    spectrum_end = time()
    
    nu_GHz = PARAMS['nu_Hz'] / 1e9
    age_Myr = PARAMS['t_age'] / 1e6
    # %g drops trailing zeros; replace '.' so it's filename-safe
    nu_str  = f"{nu_GHz:g}".replace('.', 'p')
    age_str = f"{age_Myr:g}".replace('.', 'p')
    fname = f"Cl_VV_ellmax{int(PARAMS['ell_max'])}_{nu_str}GHz_RemnAge{age_str}Myr.npz"
    np.savez(fname, ls=ls, Cl_VV=Cl_VV)

    plt.plot(ls, ls*(ls+1)*Cl_VV/(2*np.pi), 'o-', color='crimson', lw=2,) #label=r'Pop III Signal'
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$\ell(\ell+1)\ C_\ell^{VV}\ /\ 2\pi$ [$\mu K^2$]')
    plt.title(r'Circular Polarization Power Spectrum from Pop III SNe')
    plt.grid(True, which='major', alpha=0.5)
    plt.show() 

    print(f'Cosmology: {camb_end-camb_start} Faraday: {fd_end-fd_start} Spectrum: {spectrum_end-spectrum_start}')

if __name__ == "__main__":
    main()
