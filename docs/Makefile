.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV = .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

install-graphviz:
ifeq ($(shell uname), Darwin) # MacOS
	brew install graphviz
else ifeq ($(shell uname -o), Cygwin) # Windows
	choco install graphviz
else ifeq ($(shell uname -o), GNU/Linux) # Linux
	if command -v apt-get >/dev/null; then sudo apt-get install -y graphviz; fi
	if command -v yum >/dev/null; then sudo yum install -y graphviz; fi
	if command -v dnf >/dev/null; then sudo dnf install -y graphviz; fi
	if command -v pacman >/dev/null; then sudo pacman -S --noconfirm graphviz; fi
else
	@echo "Could not identify OS or appropriate package manager."
endif

.venv:  ## Set up virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements
	$(MAKE) install-graphviz

.PHONY: node_modules
node_modules:
	npm install

.PHONY: requirements
requirements: .venv
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

.PHONY: serve
serve: .venv
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && mkdocs serve

.PHONY: lint
lint: .venv node_modules
# python
	$(VENV_BIN)/black --check . 
# js
	npx rome format docs/src/node/

.PHONY: test-python
test-python: .venv
	find docs/src/ -name "*.py" | xargs -n 1 sh -c 'python $$0 || exit 255'

.PHONY: test-node
test-node: node_modules
	find docs/src/ -name "*.js" | xargs -n 1 sh -c 'node $$0 || exit 255'