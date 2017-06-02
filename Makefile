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
