read_edif gcd_hier.edn
link_design -top gcd_hier -mode out_of_context -part xc7a15tcsg325-1
opt_design -remap
place_design
report_utilization
exit
