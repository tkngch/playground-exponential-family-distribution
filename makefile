.PHONY: test

default: format lint test

test:
	python3 -m unittest discover

lint: .venv
	$</bin/mypy .
	$</bin/ruff check .

format: .venv
	$</bin/isort .
	$</bin/black .

.venv: requirements.txt
	rm -rf $@
	python3 -m venv $@
	$@/bin/pip install -r $<
