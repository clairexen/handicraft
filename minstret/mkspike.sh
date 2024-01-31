#!/bin/bash
set -ex

rm -rf riscv-fesvr
git clone https://github.com/riscv/riscv-fesvr.git riscv-fesvr
cd riscv-fesvr
# git checkout d50327f
./configure
make
cd ..

rm -rf riscv-isa-sim
git clone https://github.com/riscv/riscv-isa-sim.git riscv-isa-sim
cd riscv-isa-sim
# git checkout bed0a54
LDFLAGS="-L../riscv-fesvr" ./configure --with-isa=RV32IMC
ln -s ../riscv-fesvr/fesvr .
make
cd ..

cat > spike.sh << "EOT"
#!/bin/bash
LD_LIBRARY_PATH="./riscv-isa-sim:./riscv-fesvr" ./riscv-isa-sim/spike "$@"
EOT

chmod +x spike.sh
