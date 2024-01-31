
translate([ 0, 0, 95 ])
dxf_linear_extrude(file="lubholder.dxf", layer = "top", height=6, convexity = 5);

dxf_linear_extrude(file="lubholder.dxf", layer = "bottom", height=6, convexity = 5);

translate([ 0, 0, 10 ]) rotate([ 90, 0, 90 ])
dxf_linear_extrude(file="lubholder.dxf", layer = "side", height=6, center = true,
	origin=dxf_cross(file="lubholder.dxf", layer = "z0"), convexity = 5);

translate([ 0, 0, 10 ]) rotate([ 90, 0, 90 + 120 ])
dxf_linear_extrude(file="lubholder.dxf", layer = "side", height=6, center = true,
	origin=dxf_cross(file="lubholder.dxf", layer = "z0"), convexity = 5);

translate([ 0, 0, 10 ]) rotate([ 90, 0, 90 - 120 ])
dxf_linear_extrude(file="lubholder.dxf", layer = "side", height=6, center = true,
	origin=dxf_cross(file="lubholder.dxf", layer = "z0"), convexity = 5);
