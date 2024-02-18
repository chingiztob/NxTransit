from setuptools import find_packages, setup

setup(
    name='TransitTest',
    packages=find_packages(include=['transit']),
    install_requires=[
        "geopandas",
        "networkx",
        "osmnx",
        "pandas",
        "numpy",
        "Pympler",
        "scipy",
        "Shapely",
        "utm",
        "tqdm",
    ],
    version='0.1.4',
    description='A package for public transit routing and analysis',
    author='chingiztob',
)

#python setup.py sdist bdist_wheel

#twine upload dist/*
#twine upload --repository testpypi dist/*

#pip install --index-url https://test.pypi.org/simple/ TransitTest
#pip install --upgrade --index-url https://test.pypi.org/simple/ TransitTest

#pip install --index-url https://test.pypi.org/simple/ transit
