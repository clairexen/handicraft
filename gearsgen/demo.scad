
// dxf files generated using:
// python gearsgen.py -n 20 -m 1 -c 0 -f demo20M1C0.dxf
// python gearsgen.py -n 15 -m 1 -c 0 -f demo15M1C0.dxf

module demo001()
{
	% dxf_linear_extrude(file = "demo20M1C0.dxf", convexity = 10,
			height = 10, center = true);
	
	for (i = [-10:+10])
	assign (j = i*5.677)
	{
		rotate(j, [0, 0 ,1])
		translate([17.5, 0, i/3])
		rotate(j*20/15, [0, 0, 1])
		dxf_linear_extrude(file = "demo15M1C0.dxf", convexity = 10,
				height = 1, center = true);
	}
}

module demo002()
{
	% render(convexity = 10)
	difference() {
		cylinder(r=15, h=10, center = true);
		dxf_linear_extrude(file = "demo20M1C0.dxf", convexity = 10,
			height = 11, center = true);
	}

	for (i = [-10:+10])
	assign (j = i*5.677)
	{
		rotate(j, [0, 0, 1])
		translate([2.5, 0, i/3])
		rotate(-j*20/15, [0, 0, 1])
		dxf_linear_extrude(file = "demo15M1C0.dxf", convexity = 10,
				height = 1, center = true);
	}	
}

module demo003()
{
	# translate([-10, 0, 0])
	rotate($t*360/20, [0, 0, -1])
	dxf_linear_extrude(file = "demo20M1C0.dxf", convexity = 10,
			height = 1, center = true);

	translate([+7.5, 0, 0])
	rotate($t*360/15, [0, 0, 1])
	dxf_linear_extrude(file = "demo15M1C0.dxf", convexity = 10,
				height = 1, center = true);
}

demo001();
//demo002();
//demo003();
