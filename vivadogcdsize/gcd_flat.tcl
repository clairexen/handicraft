read_edif gcd_flat.edn
link_design -top gcd_flat -mode out_of_context -part xc7a15tcsg325-1
opt_design -remap
place_design
report_utilization
exit
