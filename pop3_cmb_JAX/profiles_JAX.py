import jax.numpy as jnp
import jax
from jax.numpy import trapezoid
jax.config.update("jax_enable_x64", True)

@jax.jit
def spherical_j1_over_x_norm(x):
    res = jnp.ones_like(x)
    mask = jnp.abs(x) < 1e-4
    safe_x = jnp.where(mask, 1.0, x)

    res = 3.0 * (jnp.sin(safe_x) - safe_x * jnp.cos(safe_x)) / safe_x**3
    return jnp.where(mask, 1.0, res)

@jax.jit
def get_SN_profile_k(k, rs, eta): # rs (,nZ)
    """
    Fourier Tansform of SN remnant shell profile, normalized to 1 at k=0.
    relativistic electrons are confined in the region rp ≤ r ≤ rs
    Faraday Conversion rate is homogeneous inside the SN remnant: u_profile(r) = 1 for rp < |r| < rs
    """
    rp = rs * (eta / (eta - 1))**(-1/3) # (,nZ)
    V_rs = (4 * jnp.pi / 3) * rs**3
    V_rp = (4 * jnp.pi / 3) * rp**3

    term1 = V_rs[:, None] * spherical_j1_over_x_norm(k[None, :] * rs[:, None])
    term2 = V_rp[:, None] * spherical_j1_over_x_norm(k[None, :] * rp[:, None])
    return (term1 - term2) / (V_rs - V_rp)[:, None]

@jax.jit
def get_halo_profile_k(k, Rvir, c_vir=10.):
    """
    Fourier Transform of Burkert halo profile, normalized to 1 at k=0.
    Burkert profile: ρ(r) = ρ0 / [(1 + r/rs) * (1 + (r/rs)^2)]
    
    c_vir: Halo Concentration Parameter
    c_vir ≤ 4 for soft halos (very sparse), c_vir ≥ 15 for compact halos (tightly packed around the center)
    """
    rs = Rvir / c_vir # (nZ, nM)
    n_r = 500 # hard coded number of radial points for integration, can be changed (accuracy/speed tradeoff)
    x = jnp.logspace(-4, 0, n_r)
    r = x[None, None, :] * Rvir[:, :, None] # (nZ, nM, nr)
    y = r / rs[:, :, None]
    
    # Burkert profile density
    rho = 1.0 / ((1 + y) * (1 + y**2))
    
    mass_shell = 4 * jnp.pi * r**2 * rho
    M_tot = trapezoid(mass_shell, x=r, axis = 2)
    
    kr = k[None, None, :, None] * r[:, :, None, :] # (nZ, nM, nK, nr)

    small = jnp.abs(kr) <= 1e-6
    safe_kr = jnp.where(small, 1.0, kr)

    sinc = jnp.sin(safe_kr) / safe_kr
    sinc = jnp.where(small, 1.0, sinc)

    integrand = mass_shell[:, :, None, :] * sinc # (nZ, nM, nK, nr)
    return trapezoid(integrand, x=r[:, :, None, :], axis=3) / M_tot[:, :, None]