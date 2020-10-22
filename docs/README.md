pyoselm's API Reference
-----------------------

This API Reference documentation is generated with [Sphinx](https://www.sphinx-doc.org/en/master/). 

## 1. Installation
You can follow the installation guide from the following [guide](http://www.sphinx-doc.org/en/master/usage/installation.html).

Basically, install the dependencies for `docs` through:

    pip install -e .[docs] 

## 2. Configuration
You can follow the [Sphinx guide](https://www.sphinx-doc.org/en/master/usage/quickstart.html).

Files to be modified will be located at `docs/source`. You will find:

 - `rst` files, index, modules, etc.
 - `conf.py` which controls how Sphinx processes your documents. 
          
## 3. Rendering
In `docs/` you will find a `Makefile`. In a bash session, you should run:

	make html
	
to render the HTML documentation from the rst files.

## 4. Hosting

To host the documentation with ReadTheDocs, follow the [official documentation](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html).