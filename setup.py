from setuptools import setup

setup(
    name="colourspace",
    version="1.0.1dev",
    packages=['colour'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'],
    author="Ivar Farup et al.",
    author_email="ivar.farup@ntnu.no",
    description="Module for colour science and colour imaging",
    license="GPL3.0",
    url="https://github.com/ifarup/colourspace"
)
