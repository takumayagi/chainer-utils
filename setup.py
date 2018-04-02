#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <signull8192@gmail.com>
#
# Distributed under terms of the MIT license.

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='chainer-utils',
    version='0.1.0',
    description='Utilities for running machine learning experiments using Chainer.',
    long_description=readme,
    author='Takuma Yagi',
    author_email='signull8192@gmail.com',
    intsll_requires=['numpy', 'chainer', 'six'],
    url='https://artilects.net',
    license=license,
    packages=find_packages(exclude=('.git', 'tests', 'docs'))
)


