default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

PYTHON_FILES=convert_ontology.py sdf/*.py setup.py
PRETTIER_FILES=.prettierrc.json *.md

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

SORT=LC_ALL=C sort --key=1,1 --key=3V --field-separator="="

reqs-fix:
	$(SORT) --output=requirements.txt requirements.txt
	$(SORT) --output=requirements-dev.txt requirements-dev.txt

reqs-check:
	$(SORT) --check requirements.txt
	$(SORT) --check requirements-dev.txt

black-fix:
	isort $(PYTHON_FILES)
	black $(PYTHON_FILES)

black-check:
	isort --check $(PYTHON_FILES)
	black --check $(PYTHON_FILES)

check: reqs-check black-check flake8 mypy lint docstyle prettier-check

precommit: reqs-fix black-fix prettier-fix check

install:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt -r requirements-dev.txt
