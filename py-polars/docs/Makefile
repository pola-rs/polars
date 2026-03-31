# Minimal makefile for Sphinx documentation

ifeq ($(shell uname),Darwin)
SPHINXOPTS    ?= -j 1 -W
else
SPHINXOPTS    ?= -j auto -W
endif
export BUILDING_SPHINX_DOCS = 1
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
INTERSPHINX   = _intersphinx

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile fetch-intersphinx

fetch-intersphinx:  ## Download intersphinx inventory files for offline/fallback use
	@mkdir -p $(INTERSPHINX)
	@test -f $(INTERSPHINX)/deltalake.inv || curl -sL https://delta-io.github.io/delta-rs/python/objects.inv -o $(INTERSPHINX)/deltalake.inv
	@test -f $(INTERSPHINX)/jax.inv || curl -sL https://jax.readthedocs.io/en/latest/objects.inv -o $(INTERSPHINX)/jax.inv
	@test -f $(INTERSPHINX)/numpy.inv || curl -sL https://numpy.org/doc/stable/objects.inv -o $(INTERSPHINX)/numpy.inv
	@test -f $(INTERSPHINX)/pandas.inv || curl -sL https://pandas.pydata.org/docs/objects.inv -o $(INTERSPHINX)/pandas.inv
	@test -f $(INTERSPHINX)/pyarrow.inv || curl -sL https://arrow.apache.org/docs/objects.inv -o $(INTERSPHINX)/pyarrow.inv
	@test -f $(INTERSPHINX)/python.inv || curl -sL https://docs.python.org/3/objects.inv -o $(INTERSPHINX)/python.inv
	@test -f $(INTERSPHINX)/sqlalchemy.inv || curl -sL https://docs.sqlalchemy.org/en/20/objects.inv -o $(INTERSPHINX)/sqlalchemy.inv
	@test -f $(INTERSPHINX)/torch.inv || curl -sL https://pytorch.org/docs/stable/objects.inv -o $(INTERSPHINX)/torch.inv

clean:
	@rm -rf source/reference/*/api/
	@rm -rf source/reference/api/
	@rm -rf "$(BUILDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: fetch-intersphinx Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
