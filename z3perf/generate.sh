#!/bin/bash
set -v
yosys -p 'write_smt2 -wires model1.smt2' model1.il
yosys -p 'write_smt2 -wires model2.smt2' model2.il
sed -r 's/rvfi_insn_(add|sub)/rvfi_insn_xxx/g; s/insn_(add|sub)\.v/insn_xxx.v/g;' < model1.smt2 > model3a.smt2
sed -r 's/rvfi_insn_(add|sub)/rvfi_insn_xxx/g; s/insn_(add|sub)\.v/insn_xxx.v/g;' < model2.smt2 > model3b.smt2
cmp model3a.smt2 model3b.smt2 || exit 1
yosys-smtbmc -t 15 --dump-smt2 test1.smt2 model1.smt2
yosys-smtbmc -t 15 --dump-smt2 test2.smt2 model2.smt2
yosys-smtbmc -t 15 --dump-smt2 test3.smt2 model3a.smt2
sed -ri '/\(set-option :produce-models true\)/ d; s/ *;.*//; /^ *$/ d;' test1.smt2
sed -ri '/\(set-option :produce-models true\)/ d; s/ *;.*//; /^ *$/ d;' test2.smt2
sed -ri '/\(set-option :produce-models true\)/ d; s/ *;.*//; /^ *$/ d;' test3.smt2
sed -r 's/rvfi_insn_(add|sub)/rvfi_insn_xxx/g;' < test1.smt2 > test1a.smt2
sed -r 's/rvfi_insn_(add|sub)/rvfi_insn_xxx/g;' < test2.smt2 > test2a.smt2
