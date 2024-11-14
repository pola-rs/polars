The documentation is split across two subfolders, `source` and `assets`.
The folder `source` contains the static source files that make up the user guide, which are mostly markdown files and the snippets of code.
The folder `assets` contains (dynamically generated) assets used by those files, including data files for the snippets and images with plots or diagrams.

Do _not_ merge the two folders together.
In [PR #18773](https://github.com/pola-rs/polars/pull/18773) we introduced this split to fix the MkDocs server live reloading.
If everything is in one folder `docs`, the MkDocs server will watch the folder `docs`.
When you make one change the MkDocs server live reloads and rebuilds the docs.
This triggers scripts that build asset files, which change the folder `docs`, leading to an infinite reloading loop.
