setMode -bs
setCable -port svf -file "example.svf"
addDevice -p 1 -file "example.jed"
Erase -p 1 
Program -p 1 -e -v 
Verify -p 1 
quit
