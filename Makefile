format:
	isort src/ --profile black
	black src/

lint:
	flake8 src/

docs:
	pydocstyle src/