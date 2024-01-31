
create_project -part xc7z010clg400-2 -in_memory

file delete -force demo004_consumer_tmp
file mkdir demo004_consumer_tmp
file copy demo004_consumer.v demo004_consumer_tmp

ipx::infer_core demo004_consumer_tmp
set_property -dict {vendor clifford.at library demo004 name consumer version 1.01} [ipx::current_core]
ipx::archive_core demo004_consumer_101.zip [ipx::current_core]

