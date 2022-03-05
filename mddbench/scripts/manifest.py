import yaml
from utils import *

with open("config.yaml") as cfg:
    config = yaml.full_load(cfg)

manifest = dict()

for name, args in config["benchmarks"].items():
    for benchmark in getBenchmarks(name, args):
        for key, val in benchmark.getManifest(config["flows"]).items():
            assert key not in manifest
            manifest[key] = val

print(yaml.dump(manifest))
