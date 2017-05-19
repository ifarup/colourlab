all: test dist

test:
	coverage run --source=colour -m unittest
	coverage html

dist:
	python3 setup.py sdist
