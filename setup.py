#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from codecs import open  # To use a consistent encoding
from os import path

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='commtailment',
    version='0.1dev',
    author="Tobias Arndt",
    author_email="tkarndt@gmail.com",
    packages=['commtailment'],
    long_description=long_description
)
