# precompute_wigner_cache.py

import os
import json
import numpy as np
import spherical


def build_wigner_cache(
    ell_min,
    ell_max,
    cache_dir="wigner_cache",
    dtype=np.float32,
):
    os.makedirs(cache_dir, exist_ok=True)

    output_ells = np.unique(
        np.logspace(np.log10(ell_min), np.log10(ell_max), 25).astype(int)
    )

    l_sum_max = ell_max
    l_sum = np.arange(2, l_sum_max)
    nL = len(output_ells)
    nl = len(l_sum)

    dtype = np.dtype(dtype)

    data_path = os.path.join(cache_dir, "wigner_kernel.dat")
    meta_path = os.path.join(cache_dir, "wigner_kernel_meta.json")

    kernel = np.memmap(
        data_path,
        dtype=dtype,
        mode="w+",
        shape=(nL, nl, nl),
    )

    kernel[:] = 0.0

    wigner_calc = spherical.Wigner3jCalculator(
        int(output_ells.max()),
        int(l_sum.max()),
    )

    for iL, L in enumerate(output_ells):
        print(f"Building cache for L={L} ({iL + 1}/{nL})")

        for idx1, l1 in enumerate(l_sum):
            l2_min = max(2, abs(int(L) - int(l1)))
            l2_max = min(l_sum_max - 1, int(L) + int(l1))

            if l2_min > l2_max:
                continue

            idx2_start = l2_min - 2
            idx2_end = l2_max - 2 + 1

            l2_vals = l_sum[idx2_start:idx2_end]

            w3j_all = wigner_calc.calculate(int(L), int(l1), 2, 0)
            w3j_vals = w3j_all[l2_vals]

            valid = ~np.isnan(w3j_vals)

            if np.any(valid):
                l2_valid = l2_vals[valid]
                idx2_valid = idx2_start + np.nonzero(valid)[0]

                weights = (
                    (2 * l1 + 1)
                    * (2 * l2_valid + 1)
                    / (4 * np.pi)
                    * w3j_vals[valid] ** 2
                )

                kernel[iL, idx1, idx2_valid] = weights.astype(dtype)

        kernel.flush()

    metadata = {
        "ell_min": int(ell_min),
        "ell_max": int(ell_max),
        "output_ells": output_ells.tolist(),
        "l_sum_start": 2,
        "l_sum_stop_exclusive": int(l_sum_max),
        "shape": [int(nL), int(nl), int(nl)],
        "dtype": str(dtype),
        "data_file": "wigner_kernel.dat",
        "description": "Dense cached coupling kernel: ((2l1+1)(2l2+1)/(4pi)) * Wigner3j^2",
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved cache to {cache_dir}")


if __name__ == "__main__":
    build_wigner_cache(
        ell_min=2,
        ell_max=3000,
        cache_dir="wigner_cache_ellmax3000",
        dtype=np.float32,
    )