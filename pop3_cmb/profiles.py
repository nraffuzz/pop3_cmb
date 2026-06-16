import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

def get_SN_profile_k(k, rs, eta):
    """
    Fourier Tansform of SN remnant shell profile, normalized to 1 at k=0.
    relativistic electrons are confined in the region rp ≤ r ≤ rs
    Faraday Conversion rate is homogeneous inside the SN remnant: u_profile(r) = 1 for rp < |r| < rs
    """
    rp = rs * (eta / (eta - 1))**(-1/3)
    V_rs = (4 * np.pi / 3) * rs**3
    V_rp = (4 * np.pi / 3) * rp**3
    
    def spherical_j1_over_x_norm(x):
        res = np.ones_like(x)
        mask = np.abs(x) > 1e-4
        if np.any(mask):
            res[mask] = 3.0 * spherical_jn(1, x[mask]) / x[mask]
        return res

    term1 = V_rs * spherical_j1_over_x_norm(k * rs)
    term2 = V_rp * spherical_j1_over_x_norm(k * rp)
    return (term1 - term2) / (V_rs - V_rp)

def get_halo_profile_k(k, Rvir, c_vir=10.):
    """
    Fourier Transform of Burkert halo profile, normalized to 1 at k=0.
    Burkert profile: ρ(r) = ρ0 / [(1 + r/rs) * (1 + (r/rs)^2)]
    
    c_vir: Halo Concentration Parameter
    c_vir ≤ 4 for soft halos (very sparse), c_vir ≥ 15 for compact halos (tightly packed around the center)
    """
    rs = Rvir / c_vir
    n_r = 500 # hard coded number of radial points for integration, can be changed (accuracy/speed tradeoff)
    x = np.logspace(-4, 0, n_r)
    r = x * Rvir
    y = r / rs
    
    # Burkert profile density
    rho = 1.0 / ((1 + y) * (1 + y**2))
    
    mass_shell = 4 * np.pi * r**2 * rho
    M_tot = simpson(mass_shell, x=r)
    
    kr = np.outer(k, r)
    sinc = np.ones_like(kr)
    mask = kr > 1e-6
    sinc[mask] = np.sin(kr[mask]) / kr[mask]
    
    integrand = mass_shell[None, :] * sinc
    return simpson(integrand, x=r, axis=1) / M_tot