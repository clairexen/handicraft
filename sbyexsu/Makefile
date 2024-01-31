TARGETS += example_pdr/example_pdr.xml
TARGETS += example_bmc3/example_bmc3.xml
TARGETS += example_yices/example_yices.xml
TARGETS += example_boolector/example_boolector.xml
# TARGETS += example_cvc4/example_cvc4.xml
TARGETS += example_cvc4x/example_cvc4x.xml

all: $(TARGETS)

example_pdr/example_pdr.xml:
	sby example.sby pdr

example_bmc3/example_bmc3.xml:
	sby example.sby bmc3

example_yices/example_yices.xml:
	sby example.sby yices

example_boolector/example_boolector.xml:
	sby example.sby boolector

example_cvc4/example_cvc4.xml:
	sby example.sby cvc4

example_cvc4x/example_cvc4x.xml:
	sby example.sby cvc4x

clean:
	rm -rf example_pdr/
	rm -rf example_bmc3/
	rm -rf example_yices/
	rm -rf example_boolector/
	rm -rf example_cvc4/
	rm -rf example_cvc4x/

.PHONY: all clean
