from setuptools import find_packages, setup

setup(
    name='TransitTest',
    packages=find_packages(include=['transit']),
    install_requires=[
        "folium",
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
    version='0.1.2',
    description='A package for public transit routing and analysis',
    author='chingiztob',
)

#python setup.py sdist bdist_wheel
#twine upload dist/*
#pip install --index-url https://test.pypi.org/simple/ transit
