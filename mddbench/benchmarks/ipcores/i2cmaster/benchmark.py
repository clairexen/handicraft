from utils import *

def getBenchmarks(name, options, args):
    return [
        Benchmark(name, options),
        Benchmark(name, options | set(["bitCtrl"])),
        Benchmark(name, options | set(["byteCtrl"]))
    ]
