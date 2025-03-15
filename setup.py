#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mediadedup",
    version="1.0.0",
    description="Intelligent media deduplication tools",
    author="Cline",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "imagehash>=4.2.0",
        "tqdm>=4.50.0",
        "colorama>=0.4.4",
    ],
    entry_points={
        'console_scripts': [
            'imagededup=mediadedup.imagededup_main:main',
            'videodedup=mediadedup.videodedup_main:main',
        ],
    },
    python_requires='>=3.7',
)
