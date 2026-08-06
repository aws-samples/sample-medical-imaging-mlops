# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'MedAnvil'
copyright = '2026, MedAnvil Team'
author = 'MedAnvil Team'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx_rtd_theme',
]

# MyST-Parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
    'tasklist',
]
myst_heading_anchors = 3
myst_links_external_new_tab = True

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'venv', '**/node_modules']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = []

# -- Options for GitHub Pages ------------------------------------------------
# When deploying to GitHub Pages under a subpath (e.g., /repo-name/),
# this is not needed if using a custom domain or deploying at root.
# html_baseurl = 'https://<your-org>.github.io/<repo-name>/'
