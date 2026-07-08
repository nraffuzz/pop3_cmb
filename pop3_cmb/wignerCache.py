import os
import json
import numpy as np


class WignerKernelCache:
    def __init__(self, cache_dir):
        meta_path = os.path.join(cache_dir, "wigner_kernel_meta.json")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.output_ells = np.array(self.meta["output_ells"], dtype=int)
        self.shape = tuple(self.meta["shape"])
        self.dtype = np.dtype(self.meta["dtype"])

        data_path = os.path.join(cache_dir, self.meta["data_file"])

        self.kernel = np.memmap(
            data_path,
            dtype=self.dtype,
            mode="r",
            shape=self.shape,
        )

    def get_kernel_for_L_index(self, iL):
        return self.kernel[iL] # recall index is shifted by 2 [2,ell_max]