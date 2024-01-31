
create_project -part xc7z010clg400-2 -in_memory

file delete -force demo006_minigpio_tmp
file mkdir demo006_minigpio_tmp
file copy demo006_minigpio.v demo006_minigpio_tmp

ipx::infer_core demo006_minigpio_tmp
set_property -dict {vendor clifford.at library demo006 name minigpio version 1.01} [ipx::current_core]
ipx::archive_core demo006_minigpio_101.zip [ipx::current_core]

