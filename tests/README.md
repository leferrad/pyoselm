# Testing

This folder contains all the tests to be handled for this library. There are two types of tests supported:
- **Unit testing:** To validate each module or block of code during development, normally done by the programmer who writes the code. Here they are implemented as simple functions executed by `pytest`.
- **System testing:** To validate the complete and fully integrated software, normally done by a tester on the completed software before it is released. Here we focus on *functional* tests, so they are written in Gherkin language and implemented with functions executed with `pytest-bdd`.

## Project structure
                            
      │
      │ ...   
      │
      ├── tests                        <- Folder for tests
      │    ├── system                  <- Core folder for System testing                         
      │    │	├── features           <- Contains .feature files where scenarios are written in Gherkin language
      │    │    │   `-- *.feature
      │    │	├── step_defs          <- Contains the definitions of the steps in Python methods     
      │    │    │   |-- __init__.py
      │    │    │   |-- utils          <- Package with utils for steps in tests    
      │    │    │   `-- conftest.py    <- Steps that are shared between scenarios                          
      │    │    │   `-- test_*.py      <- Steps for tests                                
      │    │	├── pytest.ini         <- Configuration file (e.g. to register markers) 
      │    │
      │    ├── unit                    <- Core folder for Unit testing                         
      │         `-- test_*.py│         <- Scripts for tests
      │
      │── README.md                    <- Documentation for testing

## Installation

Install the dependencies for `tests` through:

    pip install -e .[tests] 

## Execution
                                     
- To execute all the tests: 

    `pytest tests/`
    
- To execute only unit tests

    e.g: `pytest tests/unit`
    
- To execute only system tests

    e.g: `pytest tests/system`


- To execute system tests with specific markets: "pytest -m <tag> tests/system/" where tag is a tag added in the test case or .feature file

    e.g.: `pytest -m elm tests/system/`



