analyze -sv test.sv
elaborate -top {top}
clock clock
reset -expression {reset}
cover -disable *
assert -set_target_bound 22 *
prove -all -iter 22
