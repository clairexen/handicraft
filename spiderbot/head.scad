
module head(phi1 = 0, phi2 = 0)
{
	axis1 = dxf_cross(file = "leg.dxf", layer = "axis1");
	axis2 = dxf_cross(file = "leg.dxf", layer = "axis2");
	axis3 = dxf_cross(file = "leg.dxf", layer = "axis3");
	axis4 = dxf_cross(file = "leg.dxf", layer = "axis4");
	axis5 = dxf_cross(file = "leg.dxf", layer = "axis5");
	axis6 = dxf_cross(file = "leg.dxf", layer = "axis6");

	rotate(phi1) translate([ axis1[0]-axis4[0], (axis5[1]-axis4[1])/2, -21.5 ])
	{
		rotate([ 90, 0, 0 ])
		{
			rotate(phi2) translate([ -axis1[0], -axis1[1], 0])
			{
				translate([ 0, 0, -25 ])
				linear_extrude(file = "leg.dxf", layer = "head", height = 5, convexity = 2);

				translate([ 0, 0, 10 ])
				linear_extrude(file = "leg.dxf", layer = "counterhead", height = 5, convexity = 2);
			}

			rotate(phi2) translate([ -31, 0, -5]) rotate([ 0, 90, 0 ])
				linear_extrude(file = "leg.dxf", layer = "headfront",
						height = 5, origin = axis6, convexity = 2);

			translate([ -axis1[0], -axis1[1], 0]) translate([ 0, 0, 13 ])
			linear_extrude(file = "leg.dxf", layer = "bbside", height = 5, convexity = 2);

			translate([ 0, 0, -12 ])
			rotate([ 0, 180, 0 ]) servo();

			%cylinder(r = 3, h = 60, center = true);
		}
	
		translate([ 0, -13, axis2[1]-axis1[1]-5 ])
		linear_extrude(file = "leg.dxf", layer = "bbtop", origin = axis2, height = 5, convexity = 2);
	
		translate([ 0, -13, axis3[1]-axis1[1]+5 ]) rotate([ 180, 0, 0 ])
		linear_extrude(file = "leg.dxf", layer = "bbbottom", origin = axis3, height = 5, convexity = 2);

	}

	rotate(180+phi1)
	servo();

	%translate([ 0, 0, -15 ]) cylinder(r = 3, h = 60, center = true);
}

module mhead(phi1 = 0, phi2 = 0)
{
	mirror([ 0, 1, 0 ]) head(-phi1, phi2);
}

