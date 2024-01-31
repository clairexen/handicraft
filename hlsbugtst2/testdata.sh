#!/bin/bash

exec > testdata.v

echo "task testdata_pde; begin"
grep ^@@.PDE_IN runhls_prj/solution/csim/report/hls_uut_csim.log | gawk '{print "pde_data(" $3 ");"}'
echo "end endtask"

echo "task testdata_pdae; begin"
grep ^@@.PDAE_IN runhls_prj/solution/csim/report/hls_uut_csim.log | gawk '{print "pdae_data(" $3 ");"}'
echo "end endtask"

echo "task testdata_racalc; begin"
grep ^@@.RACALC runhls_prj/solution/csim/report/hls_uut_csim.log | gawk '{print "racalc_data(" $3 ");"}'
echo "end endtask"

