from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="py-polars",
    version="0.8.10",
    description="legacy version of polars",
    author="Ritchie Vink",
    author_email="ritchie46@gmail.com",
    license="MIT",
    packages=["pypolars"],
    install_requires=["polars==0.8.10"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
