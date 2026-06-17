# Environment

Last updated: `2026-06-08 09:44` KST

## Hardware

- GPU: `3 x NVIDIA Quadro RTX 6000` (`24 GB` each)
- CPU: `Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz`
- RAM: `131.6 GB` total (`125.5 GiB`)

## Software

### Main Preprocessing / Analysis Environment

Conda environment: `base`  
Python executable: `/home/hkim/miniconda3/bin/python`

- Python: `3.13.11`
- PyTorch: `2.6.0+cu124`
- CUDA used by PyTorch: `12.4`
- MNE: `1.12.0`
- MOABB: `1.5.0`
- scikit-learn: `1.8.0`
- pyriemann: `0.10`
- NumPy: `2.4.4`
- SciPy: `1.17.1`

### Training Environment

Conda environment: `mi_spdnet`  
Python executable: `/home/hkim/miniconda3/envs/mi_spdnet/bin/python`

- Python: `3.10.20`
- PyTorch: `2.6.0+cu124`
- CUDA used by PyTorch: `12.4`
- scikit-learn: `1.7.2`
- NumPy: `2.2.6`
- SciPy: `1.15.2`
- MNE: not installed
- MOABB: not installed
- pyriemann: not installed

## Notes

- Raw preprocessing, MOABB loading, MNE processing, pyriemann-based analysis, statistical tests, and visualization were run in the `base` environment.
- Deep model training runs used the `mi_spdnet` environment unless otherwise noted.
- CUDA availability was confirmed in both environments through PyTorch.
