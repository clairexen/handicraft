import yaml, os
from utils import *

with open("config.yaml") as cfg:
    config = yaml.full_load(cfg)

manifest = dict()

for name, args in config["benchmarks"].items():
    for benchmark in getBenchmarks(name, args):
        for key, val in benchmark.getManifest(config["flows"]).items():
            assert key not in manifest
            manifest[key] = val

os.makedirs("output", exist_ok=True)
with open("output/manifest.mk", "w") as f:
    for bench, tools in manifest.items():
        os.makedirs(f"output/{bench}", exist_ok=True)
        print(f"output/{bench}/benchmark.ok:", file=f)
        print(f"\t@echo 'Setting up benchmark {bench}.'", file=f)
        print(f"\t@touch output/{bench}/benchmark.ok", file=f)
        print("", file=f)

        for tool, dep in tools.items():
            os.makedirs(f"output/{bench}/datapoint-{tool}", exist_ok=True)
            if dep is None:
                dep = f"output/{bench}/benchmark.ok"
            else:
                dep = f"output/{bench}/datapoint-{dep}/datapoint.ok"
            print(f"datapoints:: output/{bench}/datapoint-{tool}/datapoint.ok", file=f)
            print(f"output/{bench}/datapoint-{tool}/datapoint.ok: {dep}", file=f)
            print(f"\t@echo 'Running benchmark {bench} through {tool}.'", file=f)
            print(f"\t@touch output/{bench}/datapoint-{tool}/datapoint.ok", file=f)
            print("", file=f)
