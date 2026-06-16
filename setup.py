from setuptools import setup, find_packages
import os

# Read requirements if you have a requirements.txt, or list them here
requirements = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "camb>=1.3.0",
    "spherical_functions",  # Ensure this library is available or point to a specific repo
    # "numba>=0.56.0",      # Uncomment if you re-introduce explicit numba JIT compilation
]

setup(
    name="pop3_cmb",
    version="0.1.0",
    author="Nicolò Raffuzzi",
    author_email="nraffuzzi@gmail.com",
    description="A framework for calculating CMB Circular Polarization (V-modes) from Population III stars via Faraday Conversion",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/nraffuzzi/pop3_cmb",
    
    # Automatically find the 'pop3_cmb' package in the directory
    packages=find_packages(),
    
    # Dependencies
    install_requires=requirements,
    
    # Python version requirement
    python_requires=">=3.8",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)