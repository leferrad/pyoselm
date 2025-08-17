Changelog
=========

Version 1.2.0 (2025)
---------------------

**New Features:**

* Modernized packaging with Poetry for better dependency management
* Added comprehensive development workflow with make commands
* Updated to support Python 3.10+

**Improvements:**

* Updated dependencies to latest versions (numpy 2.x, scipy 1.x, scikit-learn 1.x)
* Improved test suite with separate unit and system tests
* Added code quality tools (black, isort, flake8, bandit)
* Enhanced documentation build process

**Bug Fixes:**

* Fixed compatibility issues with newer scikit-learn versions
* Resolved deprecated dataset usage in tests
* Fixed online learning implementation for OS-ELM constraints

**Development:**

* Migrated from setup.py to modern pyproject.toml
* Added Poetry for dependency management
* Improved CI/CD pipeline
* Enhanced development scripts and automation

Version 1.1.x (Previous)
-------------------------

* Initial stable release
* Implementation of ELM and OS-ELM algorithms
* Scikit-learn compatible API
* Basic documentation and examples
