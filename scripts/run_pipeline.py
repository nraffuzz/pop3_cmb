import numpy as np
from pop3_cmb.config import PARAMS, CONSTANTS, COSMOLOGY
from pop3_cmb.cosmology import CAMBRunner
from pop3_cmb.faraday import FaradayModel
from pop3_cmb.spectrum import VVSpectrum

def main():
    camb_run = CAMBRunner(COSMOLOGY, PARAMS['k_min'], PARAMS['k_max'], PARAMS['n_k_points'], PARAMS['z_min'], PARAMS['z_max'])
    z_grid, k_grid, Plin = camb_run.get_linear_matter_power()
    r_grid = camb_run.get_comoving_distance(z_grid)
    ls_ee, Cl_EE = camb_run.get_primordial_Cl_EE()
    
    fm = FaradayModel(camb_run, PARAMS, CONSTANTS, k_grid, Plin)
    P_alpha = fm.compute_P_alpha()
    
    convolution = VVSpectrum(2, PARAMS['ell_max'], P_alpha, z_grid, r_grid, k_grid, Cl_EE)
    ls, Cl_VV = convolution.compute_Cl_VV()
    
    nu_GHz = PARAMS['nu_Hz'] / 1e9
    age_Myr = PARAMS['t_age'] / 1e6
    # %g drops trailing zeros; replace '.' so it's filename-safe
    nu_str  = f"{nu_GHz:g}".replace('.', 'p')
    age_str = f"{age_Myr:g}".replace('.', 'p')
    fname = f"Cl_VV_ellmax{int(PARAMS['ell_max'])}_{nu_str}GHz_RemnAge{age_str}Myr.npz"
    np.savez(fname, ls=ls, Cl_VV=Cl_VV)

if __name__ == "__main__":
    main()