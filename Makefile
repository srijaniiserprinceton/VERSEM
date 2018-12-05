# Makefile for the VERSEM Python suite
#
#

all: test

# Initializing installation
init:
	pip install -r requirements.txt

# Testing installation
test:
	py.test tests

# Make Documentation


.PHONY: init test
