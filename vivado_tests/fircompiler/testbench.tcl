
# set hz 1000
set hz 50

# set coebits 2
set coebits 14

set mode sim
# set mode syn

create_project -part xc7z010clg400-2 -force testbench testbench

create_ip -name fir_compiler -vendor xilinx.com -library ip -version 7.2 -module_name fir_compiler_0
set_property -dict "
  CONFIG.CoefficientSource {COE_File}
  CONFIG.Coefficient_File {[pwd]/fircoeff.coe}
  CONFIG.Coefficient_Width {$coebits}
  CONFIG.Filter_Type {Interpolation}
  CONFIG.Rate_Change_Type {Integer}
  CONFIG.Interpolation_Rate {4}
  CONFIG.Data_Width {22}
  CONFIG.Clock_Frequency {250.0}
" [get_ips fir_compiler_0]

if {$coebits > 2} {
  set_property CONFIG.Coefficient_Reload {true} [get_ips fir_compiler_0]
}

if {$hz == 1000} {
  # f=2.0 would be sufficient
  set_property CONFIG.Sample_Frequency {2.5} [get_ips fir_compiler_0]
}

if {$hz == 50} {
  # f=0.125 would be sufficient
  set_property CONFIG.Sample_Frequency {0.2} [get_ips fir_compiler_0]
}

add_files top.v
add_files top.xdc
add_files -fileset sim_1 testbench.v

set_property top top [current_fileset]
set_property top testbench [get_filesets sim_1]
set_property verilog_define "
  SAMPLES_IN_TXT=\"[pwd]/samples_in.txt\"
  SAMPLES_OUT_TXT=\"[pwd]/samples_out.txt\"
" [get_filesets sim_1]

generate_target all [get_files fir_compiler_0.xci]
create_ip_run [get_files fir_compiler_0.xci]
update_compile_order -force_gui

if {$mode == "sim"} {
  launch_simulation
  close_wave_config
  open_wave_config testbench.wcfg

  restart
  run 500 us
}

if {$mode == "syn"} {
  launch_runs impl_1
  wait_on_run impl_1
  open_run impl_1
  report_timing -warn_on_violation
  report_utilization
}

