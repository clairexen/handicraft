Breaking commit: [f976b16e3f2df9f7cfe0b46ecdc1cb55bdf603b6](https://github.com/Z3Prover/z3/commit/f976b16e3f2df9f7cfe0b46ecdc1cb55bdf603b6)

Checking out Z3 sources:
```
bash checkout.sh
```

Building Z3 and run test:
```
bash runtest.sh
```

Printing ccache stats from latest (or current) build:
```
bash ccachestat.sh
```

Launching a shell with ccache set up:
```
. ccachestat.sh bash
ccache -p
```

Purge ccache data and z3 git working copy:
```
bash purge.sh
```
