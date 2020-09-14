from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='rslc',
    version='0.2.0',
    description='Robust Single-Linkage Clustering',
    rust_extensions=[RustExtension(
        'rslc.rslc',
        './Cargo.toml',
        features= ["python"]
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)