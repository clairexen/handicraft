
h1 = 4;
h2 = 9.5;

difference()
{
	union() {
		linear_extrude(file = "violin_pickup.dxf", layer = "s1",
			height = h1, center = false, convexity = 10);
		linear_extrude(file = "violin_pickup.dxf", layer = "s2",
			height = h2, center = false, convexity = 10);
	}

	for (m = [1, -1])
	scale([m, 1, 1])
	intersection() {
		translate([-3, 0, 0]) rotate([0, 10, 0])
		linear_extrude(file = "violin_pickup.dxf", layer = "s2",
			height = h2*2, center = true, convexity = 10);
		translate([-50, 0, 0]) cube([70, 70, 70], center = true);
	}
}