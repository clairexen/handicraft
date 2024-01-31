// dimensions (in mm)
thickness = 0.5;
inner_dia = 15;
outer_dia = 80;
height = 100;

// other parameters
twist = 90;
spikes = 20;

module spike(inset)
{
	h = height - inner_dia/2;
	cylinder(h = h, r = inner_dia/2 - inset);
	translate([0, 0, h]) sphere(inner_dia/2 - inset);
}

module fan()
{
	render(convexity = 2) difference() {
		linear_extrude(height = height, twist = twist, $fn = 120)
			intersection() {
				translate([-thickness/2, thickness]) square([thickness, outer_dia/2]);
				rotate(120/spikes) square([outer_dia/2, outer_dia/2]);
				rotate(90-120/spikes) square([outer_dia/2, outer_dia/2]);
			}
		spike(thickness/2);
	}
}

module fan_assembly()
{
	for (i = [0:spikes-1])
		rotate([0, 0, i*360/spikes]) fan();
}

fan_assembly();
