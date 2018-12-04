# Makefile for the VERSEM Python suite
#
#



# Initializing installation
init:
	pip install -r requirements.txt

# Testing installation
test:
	py.test tests

# Make Documentation


.PHONY: init test
