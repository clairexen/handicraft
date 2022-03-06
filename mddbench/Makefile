export PYTHONPATH=$(PWD)/scripts

output/manifest.mk: scripts/manifest.py
	python3 -m manifest

clean:
	find -name __pycache__ | xargs rm -rvf
	rm -rf output
