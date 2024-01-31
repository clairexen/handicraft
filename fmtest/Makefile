all: test_boolector/.stamp test_boolector_syn/.stamp test_yices/.stamp test_yices_syn/.stamp test_abc/.stamp jgproject/jg.log

test_boolector/.stamp:
	sby test.sby boolector

test_boolector_syn/.stamp:
	sby test.sby boolector_syn

test_yices/.stamp:
	sby test.sby yices

test_yices_syn/.stamp:
	sby test.sby yices_syn

test_abc/.stamp:
	sby test.sby abc

jgproject/jg.log:
	jg -batch jgtest.tcl
