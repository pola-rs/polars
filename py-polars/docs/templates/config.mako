<%!
    # Template configuration. Copy over in your template directory
    # (used with `--template-dir`) and adapt as necessary.
    # Note, defaults are loaded from this distribution file, so your
    # config.mako only needs to contain values you want overridden.
    # You can also run pdoc with `--config KEY=VALUE` to override
    # individual values.

    html_lang = 'en'
    show_inherited_members = False
    extract_module_toc_into_sidebar = True
    list_class_variables_in_index = True
    sort_identifiers = True
    show_type_annotations = True

    # Show collapsed source code block next to each item.
    # Disabling this can improve rendering speed of large modules.
    show_source_code = True

    # If set, format links to objects in online source code repository
    # according to this template. Supported keywords for interpolation
    # are: commit, path, start_line, end_line.
    #git_link_template = 'https://github.com/USER/PROJECT/blob/{commit}/{path}#L{start_line}-L{end_line}'
    #git_link_template = 'https://gitlab.com/USER/PROJECT/blob/{commit}/{path}#L{start_line}-L{end_line}'
    #git_link_template = 'https://bitbucket.org/USER/PROJECT/src/{commit}/{path}#lines-{start_line}:{end_line}'
    #git_link_template = 'https://CGIT_HOSTNAME/PROJECT/tree/{path}?id={commit}#n{start-line}'
    git_link_template = None

    # A prefix to use for every HTML hyperlink in the generated documentation.
    # No prefix results in all links being relative.
    link_prefix = ''

    # Enable syntax highlighting for code/source blocks by including Highlight.js
    syntax_highlighting = True

    # Set the style keyword such as 'atom-one-light' or 'github-gist'
    #     Options: https://github.com/highlightjs/highlight.js/tree/master/src/styles
    #     Demo: https://highlightjs.org/static/demo/
    hljs_style = 'github'

    # If set, insert Google Analytics tracking code. Value is GA
    # tracking id (UA-XXXXXX-Y).
    google_analytics = ''

    # If set, insert Google Custom Search search bar widget above the sidebar index.
    # The whitespace-separated tokens represent arbitrary extra queries (at least one
    # must match) passed to regular Google search. Example:
    #google_search_query = 'inurl:github.com/USER/PROJECT  site:PROJECT.github.io  site:PROJECT.website'
    google_search_query = ''

    # Enable offline search using Lunr.js. For explanation of 'fuzziness' parameter, which is
    # added to every query word, see: https://lunrjs.com/guides/searching.html#fuzzy-matches
    # If 'index_docstrings' is False, a shorter index is built, indexing only
    # the full object reference names.
    #lunr_search = None
    lunr_search = {'fuzziness': 1, 'index_docstrings': True}

    # If set, render LaTeX math syntax within \(...\) (inline equations),
    # or within \[...\] or $$...$$ or `.. math::` (block equations)
    # as nicely-formatted math formulas using MathJax.
    # Note: in Python docstrings, either all backslashes need to be escaped (\\)
    # or you need to use raw r-strings.
    latex_math = True
%>
