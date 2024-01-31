#!/bin/bash
set -ex
iverilog -o quine.exe quine.v
./quine.exe | tee quine.out
diff -u quine.v quine.out
