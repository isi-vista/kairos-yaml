default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

# TODO: Remove Python distinction once everything is Blackened
NEW_PYTHON_FILES=convert_ontology.py sdf/ontology.py sdf/yaml_schema.py
PYTHON_FILES=$(NEW_PYTHON_FILES) sdf/yaml2sdf.py
YAML_FILES=.prettierrc.yaml
PRETTIER_FILES=$(YAML_FILES) *.md

PRETTIER=prettier --ignore-path .gitignore

prettier-fix:
	$(PRETTIER) --write $(PRETTIER_FILES)

prettier-check:
	$(PRETTIER) --check $(PRETTIER_FILES)

lint:
	pylint $(PYTHON_FILES)

docstyle:
	pydocstyle --convention=google $(PYTHON_FILES)

mypy:
	mypy $(PYTHON_FILES)

flake8:
	flake8 $(PYTHON_FILES)

yamllint:
	yamllint --strict $(YAML_FILES)

SORT=sort --key=1,1 --key=3V --field-separator="="

reqs-fix:
	$(SORT) --output=requirements.txt requirements.txt
	$(SORT) --output=requirements-dev.txt requirements-dev.txt

reqs-check:
	$(SORT) --check requirements.txt
	$(SORT) --check requirements-dev.txt

black-fix:
	isort $(PYTHON_FILES)
	black $(NEW_PYTHON_FILES)

black-check:
	isort --check $(PYTHON_FILES)
	black --check $(NEW_PYTHON_FILES)

check: reqs-check black-check flake8 mypy lint docstyle prettier-check yamllint

precommit: reqs-fix black-fix prettier-fix check

install:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt -r requirements-dev.txt
