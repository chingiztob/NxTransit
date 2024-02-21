from setuptools import find_packages, setup

setup(
    name='TransitTest',
    packages=find_packages(include=['transit']),
    python_requires='>=3.9',
    install_requires=[
        "geopandas>=0.14.3",
        "networkx>=3.2.1",
        "numpy>=1.26.4",
        "osmnx>=1.9.1",
        "pandas>=2.2.0",
        "Pympler>=1.0.1",
        "pytest>=8.0.1",
        "scipy>=1.12.0",
        "Shapely>=2.0.3",
        "tqdm>=4.66.1",
        "utm>=0.7.0",
    ],
    version='0.1.10',
    description='Alpha version of the package for public transit routing and analysis',
    author='chingiztob',
)

# python setup.py sdist bdist_wheel

# twine upload dist/*
# twine upload --repository testpypi dist/*

# pip install --index-url https://test.pypi.org/simple/ TransitTest
