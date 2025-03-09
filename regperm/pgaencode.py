#!/usr/bin/python3

from argparse import ArgumentParser
import pga, sys, os

opnames = """
    n z tu t tz u
    h l rh rl r
    uh tzl th tl tr
""".split()

class Cards (pga.PGA):
    def __init__ (self, args):
        self.args = args
        d = dict \
            ( maximize      = False
            , pop_size      = 26
            , num_replace   = 19
            , mutation_only = True
            , mutation_prob = 0.2
            , max_GA_iter   = 1000
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

        if not os.path.exists("pgaencode_db"):
            os.makedirs("pgaencode_db")

        if not (is_cached := os.path.isfile(f"pgaencode_db/{n}.out")):
            with open(f"pgaencode_db/{n}.v", "w") as f:
                f.write("module top (\n");
                f.write("  input [3:0] din,\n");
                f.write("  output enable_t, bypass_t,\n");
                f.write("  output enable_z, enable_u, enable_r, bypass_zur,\n");
                f.write("  output enable_l, enable_h, bypass_lh\n");
                f.write(");\n");
                for sig in "tzurlh":
                    cases = [f"din == 4'd{idx} /* {opnames[nidx]} */"
                            for idx,nidx in enumerate(g) if sig in opnames[nidx]]
                    f.write(f"  assign enable_{sig} = {' || '.join(cases)};\n");
                f.write("  assign bypass_t = !enable_t;\n");
                f.write("  assign bypass_zur = !enable_z && !enable_u && !enable_r;\n");
                f.write("  assign bypass_lh = !enable_l && !enable_h;\n");
                f.write("endmodule\n");

            ret = os.system(f"yosys -ql pgaencode_db/{n}.log -p 'synth; abc -g cmos; " +
                    f"tee -o pgaencode_db/{n}.out stat -tech cmos' pgaencode_db/{n}.v")
            assert ret == 0

        score = -1
        with open(f"pgaencode_db/{n}.out") as f:
            for line in f:
                if "Estimated number of transistors:" in line:
                    score = int(line.split()[-1])
        assert 0 < score

        print(f"{self.GA_iter:3d}{chr(65+p)}: {n} -> {score:3d} " +
                f"transistors{' (cached)' if is_cached else ''}")

        return score

    def print_string(self, file, p, pop):
        g = self.build_pheno(p, pop)
        print(" ".join([opnames[i] for i in g]), file=file)

    def stop_cond(self):
        best_idx = self.get_best_index(pga.PGA_OLDPOP)
        best_val = self.evaluate(best_idx, pga.PGA_OLDPOP)
        if best_val <= 0:
            return True
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
