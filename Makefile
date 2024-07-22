install:
	pip install --upgrade pip && pip install -r requirements.txt

#test:
	#python -m pytest -vv --cov=

format:
	@echo "Formatting all Python files"
	black *.py

lint:
	@echo "Linting all python files"
	pylint --disable=R,C *.py


refactor: format lint

all: install refactor 