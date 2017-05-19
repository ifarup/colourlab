all: test sdist

test:
	coverage run --source=colour -m unittest
	coverage html

sdist:
	python3 setup.py sdist
