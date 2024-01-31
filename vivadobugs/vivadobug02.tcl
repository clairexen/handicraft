read_verilog vivadobug02.v
synth_design -part xc7k70t-fbg676 -top top
opt_design
write_verilog -force vivadobug02_syn.v
