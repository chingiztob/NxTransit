import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
from nxtransit import _version

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NxTransit'
copyright = '2024, Chingiz Zhanarbaev'
author = 'Chingiz Zhanarbaev'

# Use the version information directly from the imported module
version = release = _version.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode",
              "myst_nb"]

templates_path = ['_templates']
exclude_patterns = []
needs_sphinx = "7.2.6"
language = "en"
source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = []
nb_execution_mode = "off"

html_theme_options = {
    "repository_provider": "github",
    "repository_url": "https://github.com/chingiztob/NxTransit",
    "use_repository_button": True,
}
html_meta = {
    "google-site-verification": "JPVWxkXKsSBRvQqXp1MfUV7TaLwUa6PlJPXV4KDEujU"
    }