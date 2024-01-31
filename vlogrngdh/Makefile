
SHELL = bash

all: demo.rpt vlogrnd01.rpt vlogrnd02.rpt vlogrnd03.rpt

demo_c: demo_c.c vlogrnd.h
	gcc -Wall -O3 -o demo_c demo_c.c

demo_v: demo_v.v
	iverilog -o demo_v demo_v.v

demo.rpt: demo_c demo_v
	( set -x; ./demo_c; ) &> demo.rpt.new
	( set -x; ./demo_v; ) &>> demo.rpt.new
	mv demo.rpt.new demo.rpt

vlogrnd01: vlogrnd01.c vlogrnd.h
	gcc -Wall -O3 -o vlogrnd01 vlogrnd01.c

vlogrnd02: vlogrnd02.c vlogrnd.h
	gcc -Wall -O3 -o vlogrnd02 vlogrnd02.c

vlogrnd03: vlogrnd03.c vlogrnd.h
	gcc -Wall -O3 -o vlogrnd03 vlogrnd03.c

vlogrnd01.rpt: vlogrnd01
	dieharder -a -g 200 < <( ./vlogrnd01; ) > >( tee vlogrnd01.rpt.new; )
	mv vlogrnd01.rpt.new vlogrnd01.rpt

vlogrnd02.rpt: vlogrnd02
	dieharder -a -g 200 < <( ./vlogrnd02; ) > >( tee vlogrnd02.rpt.new; )
	mv vlogrnd02.rpt.new vlogrnd02.rpt

vlogrnd03.rpt: vlogrnd03
	dieharder -a -g 200 < <( ./vlogrnd03; ) > >( tee vlogrnd03.rpt.new; )
	mv vlogrnd03.rpt.new vlogrnd03.rpt

