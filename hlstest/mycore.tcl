
open_project mycore_hlsprj
set_top mycore
add_files mycore.cc
add_files mycore.h

open_solution "my_solution"
set_part {xc7z030ffg676-1}
create_clock -period 6 -name default
config_bind -effort high
config_schedule -effort high

csynth_design
exit

