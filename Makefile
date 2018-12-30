# Makefile for the VERSEM Python suite
#
#

all: init test

# Initializing installation
init:
	pip install -r requirements.txt

# Testing installation
test:
	py.test tests

# Make Documentation
docs:
	cd docs
	make latexpdf
	make html

# Cleaning up
clean:
	cd docs
	make clean

.PHONY: init test
