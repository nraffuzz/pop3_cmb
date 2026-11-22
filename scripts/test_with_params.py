import numpy as np

from pop3_cmb.config import PARAMS, CONSTANTS, COSMOLOGY
from pop3_cmb.cosmology import CAMBRunner
from pop3_cmb.faraday import FaradayModel
from pop3_cmb.spectrum import VVSpectrum

import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (10, 6)


def main():
    correction = True # must be true otherwise error for this test

    nuisance = {
        'A': 1.0,
        'B': 1.0
    }

    z_re = 8.0
    Delta_z = 7

    camb_run = CAMBRunner(COSMOLOGY, PARAMS['k_min'], PARAMS['k_max'], PARAMS['n_k_points'], PARAMS['z_min'], PARAMS['z_max'])

    z_grid, k_grid, Plin = camb_run.get_linear_matter_power()
    r_grid = camb_run.get_comoving_distance(z_grid)
    ls_ee, Cl_EE = camb_run.get_primordial_Cl_EE(lmax=3000)

    fm = FaradayModel(
        camb_run,
        PARAMS,
        CONSTANTS,
        k_grid,
        Plin,

        freq=1 * 10**9,

        Model='FlatIMF',

        M_low=10,
        M_high=100,

        correction=correction,
        nuisance=nuisance,
        z_re=z_re,
        Delta_z = Delta_z
        )

    P_alpha = fm.compute_P_alpha()
    if correction: 
        tau = fm.reion.tau()

    convolution = VVSpectrum(2, PARAMS['ell_max'], P_alpha, z_grid, r_grid, k_grid, Cl_EE)
    ls, Cl_VV = convolution.compute_Cl_VV()

    nu_GHz = PARAMS['nu_Hz'] / 1e9
    age_Myr = PARAMS['t_age'] / 1e6

    nu_str = f"{nu_GHz:g}".replace('.', 'p')
    age_str = f"{age_Myr:g}".replace('.', 'p')

    fname = (
        f"Cl_VV_ellmax{int(PARAMS['ell_max'])}_"
        f"{nu_str}GHz_"
        f"RemnAge{age_str}Myr_"
        f"A{nuisance['A']}_"
        f"B{nuisance['B']}_"
        f"zre{z_re}__"
        f"DeltaZ{Delta_z}.npz"
    )

    np.savez(
        fname,
        ls=ls,
        Cl_VV=Cl_VV,
        tau=tau,
        A=nuisance['A'],
        B=nuisance['B'],
        z_re=z_re,
        Delta_z=Delta_z)

    plt.plot(ls, ls * (ls + 1) * Cl_VV / (2 * np.pi), 'o-', color='crimson', lw=2)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{VV}/2\pi$ [$\mu K^2$]')
    plt.title(f'{z_grid[-1]},{z_grid[0]}')
    plt.title(
        rf'$A={nuisance["A"]},\ B={nuisance["B"]},'
        rf'\ z_{{re}}={z_re},\ \Delta z={Delta_z:.4f},'
        rf'\ \tau={tau:.4f}$'
    )

    plt.grid(True, which='major', alpha=0.5) 
    print(z_grid)
    plt.show()

if __name__ == "__main__":
    main()