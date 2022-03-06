from utils import *

def getBenchmarks(name, options, args):
    if args == "*" and len(options) == 0:
        return [
            Benchmark(name, "fullIP"),
            Benchmark(name, "bitCtrl"),
            Benchmark(name, "byteCtrl")
        ]
    assert args is None
    return [ Benchmark(name, options) ]
