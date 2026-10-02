import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

class Reionization:

    """
    Compute the hydrogen, helium reionization fraction history and tao. 

    The following should be passed in the PARAM dict

    :param x_0: Initial reion fraction for first reion.
    :param f_He: Helium to Hydrogen fraction.
    :param z_grid: Redshift grid.
    :param delta_z: Width factor (non physical) for first reion.
    :param z_re: Redshift at Q = 0.5 for first reion.
    :param delta_z_He: Width factor (non physical) for second reion.
    :param z_re_He: Redshift at Q = 0.5 for second reion.

    The following should be passes in a nuisance dictionary  

    :param A
    :param B
    """

    def __init__(self, camb_runner, PARAM: dict, nuisance: dict, z_re, Delta_z):
        self.f_He = PARAM['f_He']
        self.x_0 = PARAM['x_0']
        self.z_re_He = PARAM['z_re_He']
        self.delta_z_He = PARAM['delta_z_He']
        self.sigma_T = PARAM['sigma_T']
        self.n_H_0 = PARAM['n_H_0']

        self.A = nuisance['A']
        self.B = nuisance['B']
        self.z_re = z_re
        self.Delta_z = Delta_z

        self.cr = camb_runner
        self.z_grid = self.cr.z_grid
        self.z_tau = np.linspace(0, 30, 1000)
        H_km_s_Mpc = self.cr.results.hubble_parameter(self.z_tau)

        Mpc_to_m = 3.085677581491367e22
        self.H_z = H_km_s_Mpc * 1000 / Mpc_to_m # (m/s/m)

        self.Q_low = 0.1
        self.Q_high = 0.9
        self.get_delta_z()

    def get_delta_z(self):
        def equation(delta_z):
            term_low = (
                (1 + self.z_re)**(3/2)
                - (3/2) * (1 + self.z_re)**(1/2)
                * delta_z
                * np.arctanh(2*self.Q_low - 1)
            )**(2/3)

            term_high = (
                (1 + self.z_re)**(3/2)
                - (3/2) * (1 + self.z_re)**(1/2)
                * delta_z
                * np.arctanh(2*self.Q_high - 1)
            )**(2/3)

            predicted_Delta_z = term_low - term_high
            return predicted_Delta_z - self.Delta_z

        delta_z = brentq(equation, 1e-6, 5) # may need to be changed if range not big enough in sampling
        self.delta_z = delta_z

    def firstReionFrac(self, filling_factor: bool = True, z = None) -> np.ndarray:
        """
        Hydrogen and first helium reionization fraction/filling factor. Hyperbolic tangent. 

        :param filling_factor: return normalized filling factor, default false
        """
        if z is None:
            z = self.z_grid

        x_full = 1 + self.f_He
        chi = ( (1 + self.z_re) ** (3/2) - (1 + z) ** (3/2) ) / ( (3/2)*(1+self.z_re) ** (1/2) * self.delta_z ) 
        x_e_Hydrogen = self.x_0 + (x_full - self.x_0) * 0.5 * (1 + np.tanh(chi))

        if filling_factor: 
            Q_HII = (x_e_Hydrogen - self.x_0) / (x_full - self.x_0)
            return Q_HII
        return x_e_Hydrogen

    def secondReionFrac(self): 
        # note: x_e_Helium -> [0,1], so it is equivlant to the filling factor
        chi = (self.z_re_He - self.z_tau) /self.delta_z_He
        self.x_e_Helium = 0.5 * (1 + np.tanh(chi))

    def fullReionFrac(self):
        self.secondReionFrac()
        x_e_Hydrogen = self.firstReionFrac(filling_factor = False, z = self.z_tau)

        self.x_e = x_e_Hydrogen + self.x_e_Helium * self.f_He

    def tau(self):
        self.fullReionFrac()
        integrand = (1+self.z_tau) ** 2 / (self.H_z) * self.x_e
        factor = 299792458 * self.sigma_T * self.n_H_0
        integral = trapezoid(integrand, x = self.z_tau)
        tao = factor * integral
        return tao

    def N_SN_cor_factor(self):
        Q_HII = self.firstReionFrac()
        cor = self.A * (1 - Q_HII) ** self.B
        return cor 