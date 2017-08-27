# Datalib

Tools to develop data-driven products. It allows to work seamlesly with data manipulation tools

Contents:

1. preprocess.py: preprocess utilities module. Use it to clean and select desired data, feature selection, drop nan values and so on
2. algorithm.py: algorithm utilities module. Use it to select best algorithm
3. algorithm_data.py: algorithm data store. It takes the base combination of algorithms and parametrizations to search for the best to fit on algorithm module with the provided data.
4. dataio.py: several input/output utilities to write results, store models (serializing them) and loading back them
5. commoncase.py: module with common cases. It exposes several utility functions for quick diagnosis of data, getting quick and dirty output from first fit algorithms and other utilities.

It also contains tests that expose the functionality of every module and function available.

