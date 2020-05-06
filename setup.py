from __future__ import absolute_import
from setuptools import setup, find_packages
import json

if __name__ == '__main__':
    with open('setup.json', 'r') as info:
        kwargs = json.load(info)
    setup(
        include_package_data=True, packages=find_packages(), setup_requires=['reentry'], reentry_register=True,
        **kwargs
    )
