![](logo.png)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MORPHӔUS is Python-based software for morphology-aware classification of individual cells and multicellular tissue structures in whole-slide multiplex images of tissue using the variational autoencoder deep learning network architecture.

## Installation
If not already installed, download `conda` following the instructions provided [here](https://docs.anaconda.com/miniconda/). Then add the `labsyspharm` channel to your conda installation:
```bash
conda config --add channels labsyspharm
```

Next, install MORPHӔUS into a dedicated Conda environment and activate it with the following commands:
```bash
conda create -n morphaeus -c conda-forge -c labsyspharm python=3.11 vae
conda activate morphaeus
```

## Program Execution
Find the `config.yml` file in the `vae` folder of the installed MORPHӔUS Conda environment and modify all applicable paths and configuration settings. Then run the program with the following command:
```bash
vae <path/to/config.yml>
```

The pipeline supports progress bookmarking, which allows the program to pick up where it left off between runs.


## MORPHӔUS Source Code

MORPHӔUS source code is freely-available for academic use and archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070212.svg)](https://doi.org/10.5281/zenodo.10070212)

---


## Funding and Acknowledgments

Supports work was supported by the Ludwig Cancer Research and the Ludwig Center at Harvard (P.K.S., S.S.) and by NIH NCI grants U54-CA225088, U2C-CA233280, and U2C-CA233262 (P.K.S., S.S.). S.S. is supported by the BWH President’s Scholars Award.

---

## References
