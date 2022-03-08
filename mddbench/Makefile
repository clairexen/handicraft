export PYTHONPATH=$(PWD)/scripts

all: datapoints

output/manifest.mk: scripts/manifest.py
	python3 -m manifest

clean:
	find -name __pycache__ | xargs rm -rvf
	rm -rf output

include output/manifest.mk

.PHONY: all datapoints clean
