# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='inventory classification',
    version='0.1.0',
    description='training and prediction job for classifying products into categories',
    long_description=readme,
    author='Gur Gosal',
    author_email='gur@dailygrabs.com',
    url='https://github.com/dailygrabsinc/ProductCategorization.git',
    license=license,
    packages=find_packages(exclude=('numpy','scikit-learn', 'pandas', 'matplotlib'))
)

