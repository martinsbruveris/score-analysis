check_dirs := score_analysis tests

# Run tests for the library
test:
	poetry run coverage erase
	poetry run coverage run -a --source=./score_analysis --branch -m pytest -s -v --black --isort tests --junit-xml unit_results.xml
	poetry run coverage report
	poetry run coverage xml
