from distutils.core import setup
from setuptools import find_packages

setup(
    name="pyoselm",
    version="1.0.1",
    author="Leandro Ferrado",
    author_email="leferrad@gmail.com",
    url="https://github.com/leferrad/pyoselm",
    packages=find_packages(exclude=['tests']),
    license="Apache License 2.0",
    description="A Python implementation of "
                "Online Sequential Extreme Machine Learning (OS-ELM) "
                "for online machine learning",
    install_requires=[
        "numpy>=1.5",
        "scipy>=0.19",
        "scikit-learn>=0.16",
    ],
    extras_require={
        'tests': [
            'pytest==6.0.1',
            'pytest-pep8==1.0.6',
            'pytest-cov==2.10.1',
            'pytest-bdd==3.4.0',
        ],
        'docs': [
            'sphinx==3.2.1',
            'sphinx-rtd-theme==0.5.0',
            'msmb_theme==1.2.0',
        ],
    }
)
