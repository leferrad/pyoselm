#! /usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

setup(
    name="pyoselm",
    version="0.1.0",
    author="Leandro Ferrado",
    author_email="ljferrado@gmail.com",
    url="https://github.com/leferrad/pyoselm",
    packages=find_packages(exclude=['examples', 'test']),
    license="LICENSE",
    description="A Python implementation of Online Sequential Extreme Machine Learning (OS-ELM) "
                "for online machine learning",
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt").read().split()
)