check_dirs := score_analysis tests

# This target runs checks on all files and potentially modifies some of them
style:
	poetry run black $(check_dirs)
	poetry run isort $(check_dirs)
	poetry run flake8 $(check_dirs)

# Run tests for the library
test:
	poetry run pytest -s -v ./tests/
