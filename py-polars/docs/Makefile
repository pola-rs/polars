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

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@rm -rf source/reference/*/api/
	@rm -rf source/reference/api/
	@rm -rf "$(BUILDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
