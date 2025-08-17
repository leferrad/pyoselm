.PHONY: help setup install install-dev test test-unit test-system lint clean build publish publish-test docs check-deps version

# Default target
help:
	@echo "Available commands:"
	@echo "  setup          Show setup instructions for development environment"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package in development mode with all dependencies"
	@echo "  test           Run unit tests with pytest"
	@echo "  test-system    Run system tests only"
	@echo "  test-unit      Run unit tests only"
	@echo "  lint           Run linting with flake8"
	@echo "  clean          Clean build artifacts and cache files"
	@echo "  build          Build package for distribution"
	@echo "  publish-test   Publish to TestPyPI"
	@echo "  publish        Publish to PyPI"
	@echo "  docs           Build documentation"
	@echo "  check-deps     Check for outdated dependencies"
	@echo "  version        Show package version"

# Setup and installation targets
setup:
	@echo "🚀 PyOSELM Development Environment Setup"
	@echo ""
	@echo "To set up the development environment:"
	@echo "  1. Install Poetry: https://python-poetry.org/docs/#installation"
	@echo "  2. Run: poetry install"
	@echo "  3. Activate shell: poetry shell"
	@echo ""
	@echo "Or use the activation script:"
	@echo "  source scripts/activate.sh"
	@echo ""
	@echo "After setup, you can use all make commands."

install:
	poetry install --only=main

install-dev:
	poetry install

# Testing targets
test:
	poetry run pytest tests/unit/ -v  --cov=pyoselm --cov-report=html --cov-report=term || true

test-unit:
	poetry run pytest tests/unit/ -v  --cov=pyoselm --cov-report=html --cov-report=term || true

test-system:
	poetry run pytest tests/system/ -v  --cov=pyoselm --cov-report=html --cov-report=term || true

# Code quality targets
lint:
	poetry run black pyoselm tests
	poetry run isort pyoselm tests
	poetry run flake8 pyoselm tests --max-line-length=88 --extend-ignore=E203,W503
	poetry run bandit -r pyoselm

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Build and publish targets
build: clean
	poetry build

publish-test: build
	poetry publish --repository testpypi

publish: build
	poetry publish

# Documentation targets
docs:
	poetry run sphinx-build -b html docs docs/_build/html
	@echo "Documentation built in docs/_build/html/"

# Dependency and security checks
check-deps:
	poetry show --outdated

# Version management
version:
	@poetry version

