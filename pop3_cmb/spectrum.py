import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from spherical_functions import Wigner3j

class VVSpectrum:
    def __init__(self, ell_min, ell_max, P_alpha, z_arr, r_arr, k_arr, Cl_EE):
        """
        Initialize the VVSpectrum object.

        Args:
            ell_min (float): min ell for output Cl_VV
            ell_max (float): max ell for output Cl_VV
            P_alpha (2D np.ndarray): Faraday Conversion power spectrum P_alpha(k, z) on the grid defined by z_arr and k_arr
            z_arr (1D np.ndarray): redshift grid
            r_arr (1D np.ndarray): radial grid
            k_arr (1D np.ndarray): wavenumber grid
            Cl_EE (1D np.ndarray): EE power spectrum on the same ell grid as Cl_VV output
        """
        self.ells = np.arange(ell_min, ell_max + 1)
        self.P_alpha = P_alpha
        self.z = z_arr
        self.r = r_arr
        self.k = k_arr
        self.Cl_EE = Cl_EE
        self.interp_P_alpha = RectBivariateSpline(self.z, np.log(self.k), np.log(self.P_alpha + 1e-30))

    def compute_Cl_alpha_limber(self, l_list):
        """
        Apply Limber approximation to compute Cl_alpha from P_alpha(k, z). 
        Returns angular power spectrum Cl_alpha on the input list of ell values.
        """
        Cl = np.zeros(len(l_list))
        for i, ell in enumerate(l_list):
            valid = self.r > 0
            r_val = self.r[valid]
            z_val = self.z[valid]
            k_req = (ell + 0.5) / r_val
            
            mask = (k_req >= self.k[0]) & (k_req <= self.k[-1])
            P_vals = np.zeros_like(k_req)
            if np.any(mask):
                log_p = self.interp_P_alpha.ev(z_val[mask], np.log(k_req[mask]))
                P_vals[mask] = np.exp(log_p)
            
            integrand = P_vals / r_val**2
            Cl[i] = np.trapz(integrand, x=r_val)
        return Cl

    def compute_Cl_VV(self):
        """
        Compute Cl_VV by convolving Cl_alpha and Cl_EE with Wigner 3j symbols:
        Equation 11 of https://arxiv.org/abs/1401.1371

        Returns:
            output_ells, Cl_VV
        """
        print("Computing Cl_VV (Convolution)...")
        t0 = time.time()

        output_ells = np.unique(np.logspace(np.log10(self.ells[0]), np.log10(self.ells[-1]), 25).astype(int))
        l_sum_max = self.ells[-1]
        l_sum = np.arange(2, l_sum_max)
        Cl_alpha = self.compute_Cl_alpha_limber(l_sum)
        
        Cl_EE_full = np.zeros(l_sum_max)
        n = min(len(self.Cl_EE), l_sum_max)
        Cl_EE_full[:n] = self.Cl_EE[:n]
        Cl_EE_cut = Cl_EE_full[l_sum]

        Cl_VV = np.zeros_like(output_ells, dtype=float)
        
        for i, L in enumerate(output_ells):
            val = 0.0
            valid_indices = np.where(Cl_alpha != 0)[0]
            for idx1 in valid_indices:
                l1 = l_sum[idx1]
                l2_min = max(2, abs(L - l1))
                l2_max = min(l_sum_max - 1, L + l1)
                if l2_min > l2_max:
                    continue
                
                idx2_start = l2_min - 2
                idx2_end = l2_max - 2 + 1
                
                l2_vals = l_sum[idx2_start:idx2_end]
                ee_vals = Cl_EE_cut[idx2_start:idx2_end]
                
                w3j_vals = np.array([Wigner3j(int(L), int(l1), int(l2), 2, 0, -2) for l2 in l2_vals])
                valid_w3j = ~np.isnan(w3j_vals)
                
                if np.any(valid_w3j):
                    weight = (2*l1 + 1) * (2*l2_vals[valid_w3j] + 1) / (4 * np.pi)
                    val += np.sum(Cl_alpha[idx1] * ee_vals[valid_w3j] * weight * w3j_vals[valid_w3j]**2)

            Cl_VV[i] = val
            print(f"  Processed ell={L}")

        minutes, seconds = divmod(time.time() - t0, 60)
        print(f"Convolution done in {int(minutes)}m {seconds:.2f}s")
        return output_ells, Cl_VV