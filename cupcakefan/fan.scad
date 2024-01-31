
bodywidth = dxf_dim(file = "fan.dxf", name = "bodywidth");
fanwidth = dxf_dim(file = "fan.dxf", name = "fanwidth");
platewidth = dxf_dim(file = "fan.dxf", name = "platewidth");
fan_side_center = dxf_cross(file = "fan.dxf",
		layer = "fan_side_center");
fanrot = dxf_dim(file = "fan.dxf", name = "fanrot");

dxf_linear_extrude(file = "fan.dxf", layer = "body",
	height = bodywidth, center = true, convexity = 10);

for (z = [+(bodywidth/2 + platewidth/2),
		-(bodywidth/2 + platewidth/2)])
{
	translate([0, 0, z])
	dxf_linear_extrude(file = "fan.dxf", layer = "plate",
		height = platewidth, center = true, convexity = 10);
}

intersection()
{
	dxf_linear_extrude(file = "fan.dxf", layer = "fan_top",
		height = fanwidth, center = true, convexity = 10,
		twist = -fanrot);
	dxf_rotate_extrude(file = "fan.dxf", layer = "fan_side",
		origin = fan_side_center, convexity = 10);
}
