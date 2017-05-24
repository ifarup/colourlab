all: test sdist

test:
	coverage run --source=colourspace -m unittest
	coverage html

sdist:
	python3 setup.py sdist

pypi: sdist
	twine upload dist/*
