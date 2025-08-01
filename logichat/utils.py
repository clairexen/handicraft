import weakref

def const(f, *, __cache__=weakref.WeakKeyDictionary()):
    try:
        return __cache__[f.__code__]
    except KeyError:
        val = f()
        __cache__[f.__code__] = val
        return val

def reload():
    exec(open("tokens.py").read(), globals())

def pr(x, *, keys=None):
    if isinstance(x, types.GeneratorType):
        x = list(x)

    if isinstance(keys, str):
        keys = keys.split()

    elif keys is None:
        if isinstance(x, str):
            print(repr(x))
            return

        if isinstance(x, (tuple, list, dict, set, frozenset)):
            if len(s := repr(x)) < 100:
                print(s)
                return
            print(s)
            return

    print(str(x))

    def val2str(x):
        if isinstance(x, (str, tuple, list, dict, set, frozenset)):
            return repr(x)
        if isinstance(x, (bytes)):
            return tuple(x)
        return x

    print("  `- " + "\n  `- ".join(f"{k:<20} {val2str(getattr(x,k))}" for k in (keys if keys else dir(x))))
