from setuptools import find_packages, setup

setup(
    name='behavioural_modelling',
    packages=find_packages(),
    version='0.1.0',
    description='A package for behavioural modelling',
    requires=['jax']
)