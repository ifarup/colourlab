from setuptools import setup

setup(
    name="colourlab",
    version="0.0.3",
    packages=['colourlab'],
    include_package_data=True,
    package_data={'colourlab': ['colour_data/*',
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
    description="Package for colour science and image processing",
    license="GPL3.0",
    url="https://github.com/ifarup/colourlab"
)
