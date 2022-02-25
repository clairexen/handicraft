#!/bin/bash
set -ex
git clone git@github.com:Z3Prover/z3.git
cd z3
git checkout -b bad f976b16e3f2df9f7cfe0b46ecdc1cb55bdf603b6
