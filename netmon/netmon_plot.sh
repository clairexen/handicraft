#!/bin/bash
gnuplot -p << EOT
set xdata time
set timefmt "%s"
set format x "%a %H:%M"
set yrange [-0.5:1.5]
plot '$1' using 1:2 with lines
EOT
