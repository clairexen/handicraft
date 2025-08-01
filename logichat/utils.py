import weakref, sys

# ======================================================================

def const(f, *, __cache__=weakref.WeakKeyDictionary()):
    try:
        return __cache__[f.__code__]
    except KeyError:
        val = f()
        __cache__[f.__code__] = val
        return val

# ======================================================================

def opts_args():
    _, *args = sys.argv
    opts = set(a for a in args if a.startswith("-"))
    args = [a for a in args if a not in opts]
    return opts, args

# ======================================================================

def embed():
    import inspect, ptpython
    caller = inspect.currentframe().f_back
    print(f"\nCalled embed() from {caller.f_code.co_filename}:{caller.f_lineno} — dropping to ptpython:")
    ptpython.repl.embed(caller.f_globals, caller.f_locals, configure=ptpy_configure)

def ptpy_configure(repl):
    if False:
        for n in dir(repl):
            if n.startswith("_"): continue
            print(n, getattr(repl, n))
    repl.swap_light_and_dark = True

# ======================================================================

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
