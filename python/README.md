# Robust Single-Linkage Clustering


## Build Instructions

To build and install the package you need to install the python dependencies:

- `numpy`
- `setuptools_rust` or `maturin`
- `pytest` (Optional, used to check if the install is working)

Then you need to have rust (cargo) installed:

- [Rust toolchain installer: rustup](https://rustup.rs/)

Build and install the package by running `python setup.py develop`.
Alternatively, if you have [maturin](https://github.com/PyO3/maturin) installed run `maturin develop --release`.

Finally check that the python package is installed and working by running `pytest`.

