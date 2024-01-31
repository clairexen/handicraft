
module outer()
{
	intersection() {
		rotate([90, 0, 0]) linear_extrude(1000, center = true, convexity = 5)
			import(file = "phonebooth_03.dxf", layer = "front");
		rotate([90, 0, 90]) linear_extrude(1000, center = true, convexity = 5)
			import(file = "phonebooth_03.dxf", layer = "front");
		linear_extrude(1000, center = true, convexity = 5)
			import(file = "phonebooth_03.dxf", layer = "top",
				origin = dxf_cross(file = "phonebooth_03.dxf", layer = "top_origin"));
	}
}

module inner()
{
	intersection() {
		rotate([90, 0, 0]) linear_extrude(1000, center = true, convexity = 5)
			import(file = "phonebooth_03.dxf", layer = "inner");
		rotate([90, 0, 90]) linear_extrude(1000, center = true, convexity = 5)
			import(file = "phonebooth_03.dxf", layer = "inner");
	}
}


difference() {
	outer();
	inner();
}