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
    version='0.1.1',
    description='My first Python library',
    author='Ch',
)

#python setup.py sdist bdist_wheel
#twine upload dist/*
#pip install --index-url https://test.pypi.org/simple/ transit
