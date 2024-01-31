
create_project -part xc7z010clg400-2 -in_memory

file delete -force demo004_generator_tmp
file mkdir demo004_generator_tmp
file copy demo004_generator.v demo004_generator_tmp

ipx::infer_core demo004_generator_tmp
set_property -dict {vendor clifford.at library demo004 name generator version 1.01} [ipx::current_core]
ipx::archive_core demo004_generator_101.zip [ipx::current_core]

