# Run tests for the library
test:
	poetry run coverage erase
	poetry run coverage run -a --source=./score_analysis --branch -m pytest -s -v tests --junit-xml unit_results.xml
	poetry run coverage report
	poetry run coverage xml
	poetry run coverage html
