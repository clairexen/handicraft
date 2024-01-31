#!/bin/bash
set -ex
mcy purge
mcy init
mcy run -j$(nproc)
