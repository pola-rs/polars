# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import inspect
import os
import re
import sys
import warnings

import sphinx_autosummary_accessors

# add polars directory
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Polars"
author = "Ritchie Vink"
copyright = f"{datetime.date.today().year}, {author}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # ----------------------
    # sphinx extensions
    # ----------------------
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # ----------------------
    # third-party extensions
    # ----------------------
    "autodocsumm",
    "numpydoc",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]  # relative to html_static_path
html_show_sourcelink = False

# adds useful copy functionality to all the examples; also
# strips the '>>>' and '...' prompt/continuation prefixes.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autosummary_generate = True
numpydoc_show_class_members = False

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
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/4UfP5cfBE7",
            "icon": "fa-brands fa-discord",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/DataPolars",
            "icon": "fa-brands fa-twitter",
        },
    ],
    "favicons": [
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "https://raw.githubusercontent.com/pola-rs/polars-static/master/icons/favicon-32x32.png",
        },
        {
            "rel": "apple-touch-icon",
            "sizes": "180x180",
            "href": "https://raw.githubusercontent.com/pola-rs/polars-static/master/icons/touchicon-180x180.png",
        },
    ],
    "logo": {
        "image_light": "https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars-logo-dark-medium.png",
        "image_dark": "https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars-logo-dimmed-medium.png",
    },
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
}


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object.

    Based on pandas equivalent:
    https://github.com/pandas-dev/pandas/blob/main/doc/source/conf.py#L629-L686
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    conf_dir_path = os.path.dirname(os.path.realpath(__file__))
    polars_root = os.path.abspath(f"{conf_dir_path}/../../polars")

    fn = os.path.relpath(fn, start=polars_root)

    return (
        f"https://github.com/pola-rs/polars/blob/master/py-polars/polars/{fn}{linespec}"
    )


def _minify_classpaths(s: str) -> str:
    # strip private polars classpaths, leaving the classname:
    # * "pli.Expr" -> "Expr"
    # * "polars.internals.expr.expr.Expr" -> "Expr"
    # * "polars.internals.lazyframe.frame.LazyFrame" -> "LazyFrame"
    # also:
    # * "datetime.date" => "date"
    s = s.replace("datetime.", "")
    return re.sub(
        pattern=r"""
        ~?
        (
          (?:pli|
            (?:polars\.
              (?:internals|datatypes)
            )
          )
          (?:\.[a-z.]+)?\.
          ([A-Z][\w.]+)
        )
        """,
        repl=r"\2",
        string=s,
        flags=re.VERBOSE,
    )


def process_signature(app, what, name, obj, opts, sig, ret):
    return (
        _minify_classpaths(sig) if sig else sig,
        _minify_classpaths(ret) if ret else ret,
    )


def setup(app):
    # TODO: a handful of methods do not seem to trigger the event for
    #  some reason (possibly @overloads?) - investigate further...
    app.connect("autodoc-process-signature", process_signature)
