from setuptools import setup, find_packages

setup(
    name='valuernn',
    version = '0.1.18',
    description = 'ValueRNN',
    author = 'Jay Hennig',
    packages = find_packages(),
    install_requires = ['matplotlib', 'numpy', 'scipy', 'scikit-learn']
)