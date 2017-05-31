test:
	coverage run --source=colourspace -m unittest
	coverage html

doc:
	sphinx-apidoc -e -f colourspace -o docs/colourspace
	cd docs && make html

sdist:
	python3 setup.py sdist

pypi: sdist
	twine upload dist/*
