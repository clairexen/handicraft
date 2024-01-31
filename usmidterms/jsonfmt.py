#!/usr/bin/env python3

import json
import sys

data = json.load(sys.stdin)
json.dump(data, sys.stdout, indent=4, sort_keys=True)
