
.include FDD8424H.lib

V1 Vcc 0 DC 
R1 Vcc 1 1

V2 VsigP 0 sin(6 5 1k)
V3 VsigN 0 sin(6 5 )

** FDD8424H Arguments: Drain Gate Source
XQ1 Vout VsigP 1 FDD8424H_Pch
XQ2 Vout VsigN 0 FDD8424H_Nch

.print tran V(Vcc) V(Vout) V(VsigN) V(VsigP) v1#branch
.tran 10.00u 5.00m 0.00m

.end

