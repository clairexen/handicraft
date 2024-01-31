#!/usr/bin/env python3

solver_list = ["cvc4", "mathsat", "yices", "z3"]

for basename in ["picorv32_async", "picorv32_sync", "ponylink_maxtx"]:
    with open(basename + ".smt2", "r") as f:
        smt2_lines = [line.strip() for line in f];

    solver_results = dict()
    for solver in solver_list:
        with open(basename + ".out_" + solver, "r") as f:
            solver_results[solver] = [line.strip() for line in f];

    with open(basename + ".smt2", "w") as f:
        check_sat_idx = 0
        for line in smt2_lines:
            if line.startswith("(set-info :status"):
                sat_list = list()
                unsat_list = list()
                for solver in solver_list:
                    if check_sat_idx < len(solver_results[solver]):
                        sat_unsat = solver_results[solver][check_sat_idx].split()[2]
                        if sat_unsat == "sat":
                            sat_list.append(solver)
                        elif sat_unsat == "unsat":
                            unsat_list.append(solver)
                        else:
                            assert False
                if len(sat_list) > 0 and len(unsat_list) == 0:
                    print("(set-info :status %s) ; SAT according to %s" % ("sat" if len(sat_list) > 1 else "unknown", " and ".join(sat_list)), file=f)
                elif len(unsat_list) > 0 and len(sat_list) == 0:
                    print("(set-info :status %s) ; UNSAT according to %s" % ("unsat" if len(unsat_list) > 1 else "unknown", " and ".join(unsat_list)), file=f)
                elif len(unsat_list) > 0 and len(sat_list) > 0:
                    print("(set-info :status unknown) ; SAT according to %s, UNSAT according to %s" % (" and ".join(sat_list), " and ".join(unsat_list)), file=f)
                else:
                    print("(set-info :status unknown)", file=f)
                check_sat_idx += 1
            else:
                print(line, file=f)

