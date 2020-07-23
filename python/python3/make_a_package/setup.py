from setuptools import setup

#  from distutils.core import setup

setup(
    name="My Test Package",
    version="0.1.0",
    author="Mladen Ivkovic",
    author_email="mladen.ivkovic@hotmail.com",
    packages=["mytestpackage"],
    license="GLPv3",
    scripts=["bin/an_example_script.py"],
    long_description=open("README.txt").read(),
    install_requires=["numpy", "matplotlib"],
)
