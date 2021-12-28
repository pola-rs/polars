# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# add polars directory
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Polars"
copyright = "2021, Ritchie Vink"
author = "Ritchie Vink"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",  # numpy docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx_panels",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]  # relative to html_static_path

html_logo = "../img/polars_logo.png"
autosummary_generate = True

html_theme_options = {
    "external_links": [
        {
            "name": "User Guide",
            "url": "https://pola-rs.github.io/polars-book/user-guide/index.html",
        },
        {"name": "Powered by Xomnia", "url": "https://www.xomnia.com/"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pola-rs/polars",
            "icon": "fab fa-github-square",
        },
    ],
}
