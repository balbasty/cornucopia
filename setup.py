#!/usr/bin/env python3
from setuptools import setup, find_packages
import versioneer

setup(
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
