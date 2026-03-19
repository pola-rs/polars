from livereload import Server, shell
from source.conf import html_static_path, templates_path

# -------------------------------------------------------------------------
# To use, just execute `python run_live_docs_server.py` in a terminal
# and a local server will run the docs in your browser, automatically
# refreshing/reloading the pages you're working on as they are modified.
# Extremely helpful to see the real output before it gets uploaded, and
# a much smoother experience than constantly running `make html` yourself.
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # establish a local docs server
    svr = Server()

    # command to rebuild the docs
    refresh_docs = shell("make html")

    # watch for source file changes and trigger rebuild/refresh
    svr.watch("*.rst", refresh_docs, delay=1)
    svr.watch("*.md", refresh_docs, delay=1)
    svr.watch("source/reference/*", refresh_docs, delay=1)
    for path in html_static_path + templates_path:
        svr.watch(f"source/{path}/*", refresh_docs, delay=1)

    # path from which to serve the docs
    svr.serve(root="build/html")
