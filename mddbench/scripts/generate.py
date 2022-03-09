import yaml, os, sys
from utils import *

def generateManifest():
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
            print(f"output/{bench}/benchmark.yaml:", file=f)
            print(f"\t@echo 'Setting up benchmark {bench}.'", file=f)
            print(f"\t$Q$(PY3) -m generate benchmark {bench}", file=f)
            print(f"\t$Qbash output/{bench}/run.sh", file=f)
            print("", file=f)

            for tool, dep in tools.items():
                os.makedirs(f"output/{bench}/datapoint-{tool}", exist_ok=True)
                if dep is None:
                    dep = f"output/{bench}/benchmark.yaml"
                else:
                    dep = f"output/{bench}/dp-{dep}/datapoint.yaml"
                print(f"datapoints:: output/{bench}/dp-{tool}/datapoint.yaml", file=f)
                print(f"output/{bench}/dp-{tool}/datapoint.yaml: {dep}", file=f)
                print(f"\t@echo 'Running benchmark {bench} through {tool}.'", file=f)
                print(f"\t$Q$(PY3) -m generate datapoint {bench} {tool}", file=f)
                print(f"\t$Qbash output/{bench}/dp-{tool}/run.sh", file=f)
                print("", file=f)

def generateBenchmark(bench):
    pass

def generateDatapoint(bench, dp):
    pass

if __name__ == "__main__":
    cmds = {
        "manifest": generateManifest,
        "benchmark": generateBenchmark,
        "datapoint": generateDatapoint
    }
    assert sys.argv[1] in cmds
    cmds[sys.argv[1]](*sys.argv[2:])
