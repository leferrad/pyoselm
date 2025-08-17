.PHONY: help install install-dev test lint format clean build publish publish-test docs check-deps all

# Default target
help:
	@echo "Available commands:"
	@echo "  setup          Show setup instructions for development environment"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package in development mode with all dependencies"
	@echo "  test           Run tests with pytest"
	@echo "  lint           Run linting with flake8"
	@echo "  clean          Clean build artifacts and cache files"
	@echo "  build          Build package for distribution"
	@echo "  publish-test   Publish to TestPyPI"
	@echo "  publish        Publish to PyPI"
	@echo "  docs           Build documentation"
	@echo "  check-deps     Check for outdated dependencies"
	@echo "  all            Run lint, test, and build"

# Setup and installation targets
setup:
	@echo "🚀 PyOSELM Development Environment Setup"
	@echo ""
	@echo "To activate the development environment, run:"
	@echo "  source scripts/activate.sh"
	@echo ""
	@echo "This will:"
	@echo "  - Create/activate a virtual environment"
	@echo "  - Install all development dependencies"
	@echo "  - Set up the project for development"
	@echo ""
	@echo "After activation, you can use all make commands."

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

# Testing targets
test:
	python -m pytest tests/ -v --cov=pyoselm --cov-report=html --cov-report=term

# Code quality targets
lint:
	python -m flake8 pyoselm tests --max-line-length=88 --extend-ignore=E203,W503
	python -m bandit -r pyoselm

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
	python -m build

publish-test: build
	python scripts/publish.py --test

publish: build
	python scripts/publish.py

# Documentation targets
docs:
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

# Dependency and security checks
check-deps:
	pip list --outdated

# Combined targets
all: lint test build

# Version management
version:
	@python -c "import pyoselm; print(f'Current version: {pyoselm.__version__}')"

