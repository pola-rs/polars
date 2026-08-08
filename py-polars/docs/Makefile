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

fetch-intersphinx:  ## Pre-fetch intersphinx inventories for slightly faster (or offline) builds
	@mkdir -p $(INTERSPHINX)
	@# Remove cached files older than 1 day so they get re-fetched
	@find $(INTERSPHINX) -name '*.inv' -mtime +1 -delete 2>/dev/null || true
	@test -s $(INTERSPHINX)/deltalake.inv || curl -sfL https://delta-io.github.io/delta-rs/python/objects.inv -o $(INTERSPHINX)/deltalake.inv || true
	@test -s $(INTERSPHINX)/jax.inv || curl -sfL https://jax.readthedocs.io/en/latest/objects.inv -o $(INTERSPHINX)/jax.inv || true
	@test -s $(INTERSPHINX)/numpy.inv || curl -sfL https://numpy.org/doc/stable/objects.inv -o $(INTERSPHINX)/numpy.inv || true
	@test -s $(INTERSPHINX)/pandas.inv || curl -sfL https://pandas.pydata.org/docs/objects.inv -o $(INTERSPHINX)/pandas.inv || true
	@test -s $(INTERSPHINX)/pyarrow.inv || curl -sfL https://arrow.apache.org/docs/objects.inv -o $(INTERSPHINX)/pyarrow.inv || true
	@test -s $(INTERSPHINX)/python.inv || curl -sfL https://docs.python.org/3/objects.inv -o $(INTERSPHINX)/python.inv || true
	@test -s $(INTERSPHINX)/sqlalchemy.inv || curl -sfL https://docs.sqlalchemy.org/en/20/objects.inv -o $(INTERSPHINX)/sqlalchemy.inv || true
	@test -s $(INTERSPHINX)/torch.inv || curl -sfL https://pytorch.org/docs/stable/objects.inv -o $(INTERSPHINX)/torch.inv || true

clean:
	@rm -rf source/reference/*/api/
	@rm -rf source/reference/api/
	@rm -rf "$(BUILDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
