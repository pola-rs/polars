#!/bin/bash

python setup.py sdist
twine upload -r pypi dist/*
rm -r dist py_polars.*
