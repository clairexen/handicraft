
module body()
{
	difference()
	{
		// a round box
		intersection() {
			hull() for (x = [-100, +100], y = [-100, +100], z = [-20, 470])
				translate([x, y, z]) sphere(20);
			translate([0, 0, 250]) cube(500, center=true);
		}

		// inside
		cube([200, 200, 900], center = true);

		// free space at the bottom
		cube([300, 200, 50], center = true);
		cube([200, 300, 50], center = true);
	}
}

module window_stencil()
{
	for (dim = [ [300, 150, 150], [150, 300, 150] ], z = [ 150, 350 ])
		translate([ 0, 0, z ]) cube(dim, center = true);
}

module booth_metal()
{
	render(convexity = 10)
	difference()
	{
		body();
		window_stencil();
	}
}

module booth_glass()
{
	render(convexity = 10)
	intersection()
	{
		body();
		window_stencil();
	}
}

color("gray") booth_metal();
color("blue", 0.2) booth_glass();

