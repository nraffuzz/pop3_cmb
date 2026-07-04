# --- Physical Constants & Reference Values ---
CONSTANTS = {
    'M_sun_kg': 1.988416e30,       # Solar mass in kg
    'pc_to_m': 3.085677581e16,     # Parsec in meters
    'Mpc_to_pc': 1e6,              # Megaparsec to parsec
    'rho_crit0_ref': 2.775e11,     # Reference critical density h^2 M_sun/Mpc^3
    
    # Reference values for scaling relations
    'Tvir_ref': 1e4,    # [K]
    'mu_ref': 0.6,      # Mean molecular weight
    'E_sn_ref': 1e53,   # [erg] Reference SN energy
    'nu_Hz_ref': 1e9,   # [Hz] Reference frequency (1 GHz)
    'fB_ref': 0.1,      # Magnetic field efficiency
    'frel_ref': 0.1,    # Relativistic electron efficiency
    't_age_ref': 1e6,   # [yr] Reference age
    'ombh2_ref': 0.02237
}

# --- Cosmological & CAMB Parameters ---
COSMOLOGY = {
    'H0': 67.5,
    'ombh2': 0.02237,
    'omch2': 0.1200,
    'omk': 0,
    'tau': 0.0544,
    'As': 2.0989e-9,
    'ns': 0.9646,
    'r': 0,
    'nt': 0,
    'ntrun': 0,
    'rho_crit0': 2.775e11, # Critical density at z=0 h^2 M_sun/Mpc^3

}

# --- Model Parameters ---
PARAMS = {
    'ell_max': 3000,
    # Redshift range for integration
    'z_min': 5,
    'z_max': 30,
    
    # Wavenumber grid [1/Mpc]
    'k_min': 1e-4,
    'k_max': 1e3,
    'n_k_points': 500,

    # Halo Mass Function (Sheth-Tormen)
    'st_a': 0.75,
    'st_p': 0.3,
    'st_A': 0.3222,

    # Pop III Star Formation & SN Physics
    'epsilon_star': 0.1,   # Star formation efficiency
    'M_sn_min': 10.0,      # [M_sun] Min progenitor mass
    'M_sn_max': 40.0,      # [M_sun] Max progenitor mass
    'E_sn': 1e51,          # [erg] SN Energy
    'eta': 4,              # Compression ratio
    'fB': 0.1,             # Magnetic field efficiency
    'frel': 0.1,           # Relativistic electron efficiency
    't_age': 1e4,          # [yr] Remnant age
    'nu_Hz': 1e9,          # [Hz] Observation frequency
    
    # Numerical Integration Grids
    'n_M_points': 100,     # Mass integration steps
    'M_halo_min': 1e5,     # [M_sun] Grid min
    'M_halo_max': 1e10,    # [M_sun] Grid max
}