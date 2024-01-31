open_project demo005_hlsconsumer_tmp
add_files demo005_hlsconsumer.c
open_solution sol_1
set_part xc7z010clg400-2
set_top hlsconsumer
csynth_design
export_design -vendor clifford.at -library demo005 -version 1.01
exit
