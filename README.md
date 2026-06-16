# pop3_cmb_pol

A Python framework for calculating Cosmic Microwave Background (CMB) Circular Polarization (V-modes) originating from Population III stars via Faraday Conversion.

This package computes the 3D Faraday Conversion Power Spectrum ($P_\\alpha$) from Pop III supernova remnants, applies the Limber approximation to get the 2D angular spectrum ($C_\\ell^\\alpha$), and convolves it with the primordial E-mode spectrum ($C_\\ell^{EE}$) to generate the final V-mode circular polarization spectrum ($C_\\ell^{VV}$).

## Installation

It is recommended to install this package inside a virtual environment (like `conda` or `venv`). 

First, clone the repository to your local machine:
```bash
git clone [https://github.com/nraffuzzi/pop3_cmb.git](https://github.com/nraffuzzi/pop3_cmb.git)
cd pop3_cmb