# Run tests for the library
test:
	uv run coverage erase
	uv run coverage run -a --source=./score_analysis --branch -m pytest -s -v tests --junit-xml unit_results.xml
	uv run coverage report
	uv run coverage xml
	uv run coverage html
