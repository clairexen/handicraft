from utils import *

def getBenchmarks(name, options, args):
    if args == "*":
        return [
            Benchmark(name, options | set(["fullIP"])),
            Benchmark(name, options | set(["bitCtrl"])),
            Benchmark(name, options | set(["byteCtrl"]))
        ]
    return [ Benchmark(name, options) ]
