ARGS ?=
POETRY ?= poetry

.PHONY: all install build clean lint format test help

all: build


install:
	$(POETRY) install --all-extras

build: install
	$(POETRY) build

clean:
	rm -rf $$($(POETRY) env info --path)
	rm -rf .nox
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name __pycache__ | xargs -r rm -r

publish: build
	$(POETRY) publish -u __token__ -p $(PYPI_TOKEN) --skip-existing

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test $(if $(PYTHON),--python=$(PYTHON),) -- $(ARGS)

help:
	@echo '===================='
	@echo 'build                        - build the library'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'publish                      - publish the library to Pypi'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test PYTHON=<python_version> - run unit tests with the specific python version'
	@echo 'test PYTHON=3.11 ARGS="--llm-mode=real"  - run unit tests with the the real LLM'
