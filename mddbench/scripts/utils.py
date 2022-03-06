class Benchmark:
    def __init__(self, name, options):
        if type(options) is str:
            self.options = set(options.split("+"))
        else:
            self.options = set(options)

        if "" in self.options:
            self.options.remove("")

        self.name = name
        self.fullName = "+".join([name] + sorted(self.options))

    def getManifest(self, flows):
        manifest = dict()
        for f in flows:
            n = None
            for t in f:
                if n is None:
                    d = None
                    n = t
                else:
                    d = n
                    n = n + "++" + t
                manifest[n] = d
        return { self.fullName: manifest };

def getBenchmark(name):
    benchmarks = getBenchmarks(name, None)
    assert len(benchmarks) == 1
    return benchmarks[0]

def getBenchmarks(name, args):
    nameParts = name.split("+")
    mod = __import__("benchmarks." + nameParts[0].replace("-", ".") + ".benchmark",
            None, None, ["getBenchmark"])
    return mod.getBenchmarks(nameParts[0], set(nameParts[1:]), args)
