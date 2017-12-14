default: test

test:
	coverage run --source=colourlab -m unittest
	coverage html

doc:
	sphinx-apidoc -e -f colourlab -o docs/colourlab
	cd docs && make html

sdist:
	python3 setup.py sdist

pypi: sdist
	twine upload dist/*

clean:
	rm -rf colourlab.egg-info
	rm -rf dist
	rm -rf htmlcov
	find . -iname '*.pyc' | xargs rm
	find . -iname '*__pycache__*' | xargs rm -rf
	cd docs && make clean

all: clean test doc sdist
