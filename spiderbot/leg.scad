
module leg(phi1 = 0, phi2 = 0, phi3 = 0)
{
	axis1 = dxf_cross(file = "leg.dxf", layer = "axis1");
	axis2 = dxf_cross(file = "leg.dxf", layer = "axis2");
	axis3 = dxf_cross(file = "leg.dxf", layer = "axis3");
	axis4 = dxf_cross(file = "leg.dxf", layer = "axis4");
	axis5 = dxf_cross(file = "leg.dxf", layer = "axis5");

	rotate(phi1) translate([ axis1[0]-axis4[0], (axis5[1]-axis4[1])/2, -21.5 ])
	{
		rotate([ 90, 0, 0 ])
		{
			rotate(phi2) translate([ -axis1[0], -axis1[1], 0])
			{
				translate([ 0, 0, -7 ]) rotate(-phi3)
				linear_extrude(file = "leg.dxf", layer = "pin", height = 5, convexity = 2);
				
				translate([ 0, 0, 12 ]) rotate(-phi3)
				linear_extrude(file = "leg.dxf", layer = "counterpin", height = 5, convexity = 2);
				
				for (off = [20, -25]) translate([ 0, 0, off ])
				linear_extrude(file = "leg.dxf", layer = "midpart", height = 5, convexity = 2);

				translate([ 0, 0, -13 ]) rotate([0, 180, -17-phi3]) servo();

				%cylinder(r = 3, h = 60, center = true);
			}
			
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

module mleg(phi1 = 0, phi2 = 0, phi3 = 0)
{
	mirror([ 0, 1, 0 ]) leg(-phi1, phi2, phi3);
}

