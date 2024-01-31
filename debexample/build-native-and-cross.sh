#!/bin/bash

set -ex
cd exampleproject

# native build
debuild -us -uc -b

# cross build for armhf
debuild -aarmhf -us -uc -B

