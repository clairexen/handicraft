
module fragile_coupler(diameter = 20, teeth = 20, length = 20, hole = 3, offset = 0.5, steps = 72)
{
	module singletooth(z1, z2, angle)
	{
		eps = 1.0;
		angle_eps = 0.1;

		x1 = diameter/2+eps;
		x2 = (diameter/2+eps)*cos(angle+angle_eps);
		y2 = (diameter/2+eps)*sin(angle+angle_eps);

		polyhedron(
			points = [
				/* 0 */ [0, 0, 0],
				/* 1 */ [0, 0, teeth],
				/* 2 */ [x1, 0, z1],
				/* 3 */ [x1, 0, teeth],
				/* 4 */ [x2, y2, z2],
				/* 5 */ [x2, y2, teeth]
			],
			triangles = [
				[1, 3, 2, 0],
				[2, 3, 5, 4],
				[0, 4, 5, 1],
				[0, 2, 4],
				[1, 5, 3]
			]
		);
	}

	module teethmask()
	{
		for (step = [0:steps-1])
			rotate([0, 0, step*360/steps])
				singletooth(cos(step*360/steps*3)*teeth/2, cos((step+1)*360/steps*3)*teeth/2, 360/steps);
	}

	render(convexity = 3) difference() {
		translate([0, 0, -length])
			cylinder(r = diameter/2, h = length + teeth/2, $fn = steps);
		translate([0, 0, offset/2]) rotate([0, 90, 0])
			cylinder(r = hole/2, h = diameter*2, center = true, $fn = 20);
		teethmask();
	}
}

module fragile_coupler_demo()
{
	diameter = 20;
	teeth = 15;
	length = 30;
	hole = 2.0;
	offset = 1.0;
	steps = 36;

	for (i = [-1:2:1]) translate([0, i*diameter*0.6, 0]) {
		color([0.2, 0.8, 0.8])
			translate([0, 0, -offset/2])
				fragile_coupler(diameter, teeth, length, hole, offset);
		color([0.8, 0.8, 0.2]) if (i < 0)
			translate([0, 0, +offset/2]) rotate([0, 180, 0])
				fragile_coupler(diameter, teeth, length, hole, offset);
		color([0.8, 0.2, 0.2])
			rotate([0, 90, 0]) cylinder(r = hole/2.2, h = diameter*2, center = true, $fn = 20);
	}
}

fragile_coupler_demo();

