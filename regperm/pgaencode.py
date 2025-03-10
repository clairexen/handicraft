#!/usr/bin/python3

from argparse import ArgumentParser
import pga, sys, os

#tech = "cmos"
#tech = "lut3"
tech = "gates"
show = True

opnames = """
    n t rl tr
    h th rh r
    tz tu tzl uh
    l tl z u
""".split()

class Cards (pga.PGA):
    def __init__ (self, args):
        self.args = args
        d = dict \
            ( maximize      = False
            , pop_size      = 26
            , num_replace   = 22
            , mutation_only = True
            , mutation_prob = 0.2
            , max_GA_iter   = 200
            , print_options = [pga.PGA_REPORT_STRING]
            , random_seed   = self.args.random_seed
            )
        if self.args.output_file:
            d ['output_file'] = self.args.output_file
        super (self.__class__, self).__init__ (int, 16, **d)

    def build_pheno(self, p, pop):
        g = []
        for i in range(len(self)):
            g.append(self.get_allele(p, pop, i))
        return g

    def evaluate(self, p, pop):
        g = self.build_pheno(p, pop)
        n = '_'.join([opnames[i] for i in g])
        nn = f"{n}.{tech}"

        if not os.path.exists("pgaencode_db"):
            os.makedirs("pgaencode_db")

        if not (is_cached := os.path.isfile(f"pgaencode_db/{nn}.out")):
            with open(f"pgaencode_db/{n}.v", "w") as f:
                f.write("module top (\n");
                f.write("  input [3:0] din,\n");
                f.write("  output enable_t,\n");
                f.write("  output enable_z, enable_u, enable_r,\n");
                f.write("  output enable_l, enable_h\n");
                f.write(");\n");
                for sig in "tzurlh":
                    cases = [f"din == 4'd{idx} /* {opnames[nidx]} */"
                            for idx,nidx in enumerate(g) if sig in opnames[nidx]]
                    f.write(f"  assign enable_{sig} = {' || '.join(cases)};\n");
                f.write("endmodule\n");

            match tech:
                case "cmos":
                    ret = os.system(f"yosys -ql pgaencode_db/{nn}.log -p 'synth; abc -g cmos; " +
                            f"tee -o pgaencode_db/{nn}.out stat -tech cmos' pgaencode_db/{n}.v")
                case "lut3":
                    ret = os.system(f"yosys -ql pgaencode_db/{nn}.log " +
                            f"-p 'synth; abc -luts 2,3,5; techmap -map lutmap.v; " +
                            f"tee -o pgaencode_db/{nn}.out stat' lutlib.v pgaencode_db/{n}.v")
                case "gates":
                    ret = os.system(f"yosys -ql pgaencode_db/{nn}.log -p 'synth; abc -g gates; " +
                            f"tee -o pgaencode_db/{nn}.out stat' pgaencode_db/{n}.v")
                case _:
                    raise AssertionError
            assert ret == 0

        score = None
        with open(f"pgaencode_db/{nn}.out") as f:
            lut1_cnt, lut2_cnt, lut3_cnt = 0, 0, 0
            for line in f:
                if "Number of cells:" in line:
                    score = (int(line.split()[-1]), "cells")
                if "Estimated number of transistors:" in line:
                    score = (int(line.split()[-1]), "transistors")
                if "LUT1" in line:
                    lut1_cnt = int(line.split()[-1])
                if "LUT2" in line:
                    lut2_cnt = int(line.split()[-1])
                if "LUT3" in line:
                    lut3_cnt = int(line.split()[-1])
            if lut1_cnt or lut2_cnt or lut3_cnt:
                score = (2*lut1_cnt + 3*lut2_cnt + 5*lut3_cnt, "pseudo-LUTs")

        assert score is not None

        if True:
            print(f"{self.GA_iter:5d}-{chr(65+p)}: {n} -> {score[0]:3d} " +
                    f"{score[1]}{' (cached)' if is_cached else ''}")
            if p == 25: print()

        return score[0]

    def print_string(self, file, p, pop):
        g = self.build_pheno(p, pop)
        n = '_'.join([opnames[i] for i in g])
        if show:
            match tech:
                case "cmos":
                    os.system(f"yosys -qp 'synth; abc -g cmos; splitnets -ports; " +
                            f"clean -purge; show -stretch top' pgaencode_db/{n}.v")
                case "lut3":
                    os.system(f"yosys -qp 'synth; abc -luts 2,3,5; techmap -map lutmap.v; splitnets -ports; " +
                            f"clean -purge; show -stretch top' lutlib.v pgaencode_db/{n}.v")
                case "gates":
                    os.system(f"yosys -qp 'synth; abc -g gates; splitnets -ports; " +
                            f"clean -purge; show -stretch top' pgaencode_db/{n}.v")
                case _:
                    raise AssertionError
        print(n, file=file)

    def stop_cond(self):
        best_idx = self.get_best_index(pga.PGA_OLDPOP)
        best_val = self.evaluate(best_idx, pga.PGA_OLDPOP)
        if best_val <= 0: return True
        return self.check_stopping_conditions ()

def main(argv):
    cmd = ArgumentParser()
    cmd.add_argument("-O", "--output-file",
            help="Output file for progress information")
    cmd.add_argument("-R", "--random-seed", type=int, default = 42,
            help="Seed random number generator, default=%(default)s")
    args = cmd.parse_args(argv)
    pg = Cards(args)
    pg.run()

if __name__ == '__main__':
    main (sys.argv[1:])
