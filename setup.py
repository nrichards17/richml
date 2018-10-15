#! /usr/bin/env python
#
# Copyright (c) 2018 Nathaniel Richards

DESCRIPTION = ''        # TODO
LONG_DESCRIPTION = ''   # TODO

DISTNAME = 'richml'
MAINTAINER = 'Nathaniel Richards'
MAINTAINER_EMAIL = 'nathaniel.richards17@gmail.com'
LICENSE = 'MIT License'
DOWNLOAD_URL = ''       # TODO

VERSION = '0.1.0'

INSTALL_REQUIRES = [
    'numpy>=1.15.0',
    'torch>=0.4.1',
]

PACKAGES = [
    'richml',
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        # url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
    )