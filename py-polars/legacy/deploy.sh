#!/bin/bash

python setup.py sdist
twine upload sdist/*
rm -r dist py_polars.*
