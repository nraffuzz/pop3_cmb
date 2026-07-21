import numpy as np
from typing import Optional 
from scipy.integrate import trapezoid

def IMF(model, M_star, M_sn_max, M_sn_min, M_low, M_high, 
        gamma: Optional[float] = None,
        alpha: Optional[float] = None, 
        beta: Optional[float] = None,
        Mch: Optional[float] = None, 
        sigma: Optional[float] = None):
    """
    Compute the N_sn per halo based on the specified model 

    Args:
        model : IMF model
        - PowerIMF : power law IMF with free gamma param
        - FlatIMF : power law IMF with gamma = 0 
        - LogNormalIMF : Log normal IMF with Mch and sigma as free param
        - ChabrierIMF : Chabrier IMF with free alpha and beta param
        - LarsonIMF : Chabrier IMF with alpha = -2.35 and beta = 1

        M_star (np.arry) : amount of stellar mass in halo of a range of masses
        M_sn_max (float) : maximum stellar mass for CCSN
        M_sn_min (float) : minimum stellar mass for CCSN 
        M_low (float) : minimum stellar mass of population 
        M_high (float) : maximum stellar mass of population 

        Note: Gamma, Sigma, and Mch is defined positive 
    """

    M_points = 500 # hard coded
    starM_grid = np.linspace(M_low, M_high, M_points)

    if model == 'PowerIMF' or model == 'FlatIMF':
        # General Power IMF
        if model =='FlatIMF':
            gamma = 0
        # Gamma != None
        elif gamma is None:
            raise ValueError("PowerIMF requires gamma.")
        phi = starM_grid ** - (gamma + 1) 

    elif model == 'LogNormalIMF':
        # Log-Normal IMF
        # Mch != None 
        # sigma != None
        if Mch is None or sigma is None:
            raise ValueError("LogNormalIMF requires Mch and sigma")
        phi = (1 / starM_grid) * np.exp(-((np.log10(starM_grid / Mch))/ (np.sqrt(2) * sigma)) ** 2)

    elif model == 'ChabrierIMF' or model == 'LarsonIMF':
        # General Chabrier IMF 
        if model == 'LarsonIMF':
            # larson IMF

            alpha = 2.35
            beta = 1
            Mch = 20
        
        # alpha != None
        # beta != None
        # Mch != None
        if alpha is None or beta is None or Mch is None:
            raise ValueError("ChabrierIMF requires alpha beta and Mch")
        phi = starM_grid ** - alpha * np.exp(-(Mch/starM_grid) ** beta)

    A_IMF = M_star / trapezoid(starM_grid * phi, x = starM_grid)
    max_idx = np.argmin(np.abs(starM_grid - M_sn_max)) 
    min_idx = np.argmin(np.abs(starM_grid - M_sn_min))
    starM_grid_sn = starM_grid[min_idx:max_idx + 1]
    phi_conf = phi[min_idx:max_idx + 1]
    N_sn_val = A_IMF * trapezoid(phi_conf, x = starM_grid_sn)

    return N_sn_val