This directory contains example SMT-LIB 2.6 code for a proposed encoding of state machines in
the SMT-LIB 2.6 language. If this representation is deemed useful, we might add support to
Yosys for generating this format from Verilog circuit descriptions.

Data types for a top-level module "design":

```
design-init	All uninitialized registers
design-input	All top-level input signals (and unconstrained internal wires)
design-state	All registers
design-values	All wires in the design
```

Functions for top-level module “design”

```
(design-first design-init)			⟶	design-state
(design-eval design-state design-inputs)	⟶	design-values
(design-next design-values)			⟶	design-state
```

Additional functions for checkign formal properties (not included in this example):

```
(design-asserts design-values)	⟶	bitmask
(design-assumes design-values)	⟶	bitmask
(design-covers design-values)	⟶	bitmask
```

Additional recursive data types and recursive functions for representing entire traces:

```
Constructors:
(design-trace-first design-input design-init)	⟶	design-trace
(design-trace-next design-input design-trace)	⟶	design-trace

Functions:
(design-trace-eval design-trace)	⟶	design-values
(design-trace-depth design-trace)	⟶	Int
(design-trace-at design-trace Int)	⟶	trace
```

The ``(design-trace-eval ...)`` function returns the design-values data
structure for the final state of the design, i.e. the values at the end of the
trace.

The ``(design-trace-at ...)`` function shapes the given number of cycles off
the end of the trace. This means, for the purpose of this function, time is
measured from the end of the trace, and 0 denotes the final state, 1 the cycle
before the final state, and so forth.

The file [counter.v](counter.v) contains the design we are using in the example,
and [counter.smt26](counter.smt26) contains SMT code for finding a trace of depth 5,
with counter initialized to 7 and counter=2 in the last cycle.

The file [trace.smt26](trace.smt26) demonstrates the additional functions for
representing entire traces as recursive data structures. It is based on a modified
version of the counter example, that has counter=16 set as initialization value
in the design itself, and is looking for a trace that ends in counter=7.
