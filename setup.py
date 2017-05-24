from setuptools import setup

setup(
    name="colourspace",
    version="1.1.0.dev",
    packages=['colourspace'],
    include_package_data=True,
    package_data={'colourspace': ['colour_data/*',
                                  'tensor_data/*',
                                  'metric_data/*']},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numba',
        'sphinxcontrib-napoleon'
    ],
    author="Ivar Farup et al.",
    author_email="ivar.farup@ntnu.no",
    description="Module for colour science and colour imaging",
    license="GPL3.0",
    url="https://github.com/ifarup/colourspace"
)
