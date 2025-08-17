Setup
=====

``pyoselm`` supports Python 3.10+.

Installing
----------

It is recommended to install this library via pip:

.. code-block:: bash

    $ pip install pyoselm

You can also install the development version from master branch of Git repository:

.. code-block:: bash

    $ pip install git+https://github.com/leferrad/pyoselm.git


Development Setup
-----------------

For development, this project uses `Poetry <https://python-poetry.org/>`_ for dependency management:

.. code-block:: bash

    # Install Poetry
    curl -sSL https://install.python-poetry.org | python3 -

    # Install dependencies
    poetry install

    # Activate environment
    poetry shell


Testing
-------

Tests are developed using `pytest <https://docs.pytest.org/en/stable/>`_ and its plugins.

To run tests, use the provided make commands:

.. code-block:: bash

    make test           # Unit tests
    make test-system    # System tests
    make lint           # Code quality checks

Or run pytest directly:

.. code-block:: bash

    poetry run pytest tests/unit/ -v

