# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'MedML Training Dashboard'
copyright = '2026, MedML Team'
author = 'MedML Team'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
]

# Try to add sphinx_rtd_theme extension if available
try:
    import sphinx_rtd_theme
    extensions.append('sphinx_rtd_theme')
except ImportError:
    pass

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'venv', '**/node_modules']

# -- Options for HTML output -------------------------------------------------

try:
    import sphinx_rtd_theme  # noqa: F811
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

html_static_path = []
