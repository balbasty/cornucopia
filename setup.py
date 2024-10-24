#!/usr/bin/env python3
import os
import sys
from setuptools import setup

# The following versioneer hack copied from dandi-cli
# https://github.com/dandi/dandi-cli/blob/master/setup.py


# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
sys.path.insert(0, os.path.dirname(__file__))

try:
    import versioneer

    setup_kw = {
        "version": versioneer.get_version(),
        "cmdclass": versioneer.get_cmdclass(),
    }
except ImportError:
    # see https://github.com/warner/python-versioneer/issues/192
    print("WARNING: failed to import versioneer, falling back to no version for now")
    setup_kw = {}

setup(**setup_kw)
