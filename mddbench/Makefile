PY3=PYTHONPATH=$(PWD)/scripts python3
Q=

all: datapoints

static:
	$(MAKE) -C benchmarks/ipcores/i2cmaster

output/manifest.mk: scripts/generate.py
	$(PY3) -m generate manifest

clean:
	find -name __pycache__ | xargs rm -rvf
	rm -rf output

include output/manifest.mk

.PHONY: all datapoints clean
