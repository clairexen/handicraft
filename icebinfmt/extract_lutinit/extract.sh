#!/bin/bash

set -ex
make -C ../utils
../utils/extractor -d binfiles > decls.txt

