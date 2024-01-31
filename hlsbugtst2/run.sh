#!/bin/bash
set -ex
bash runhls.sh
bash testdata.sh
bash testbench.sh
