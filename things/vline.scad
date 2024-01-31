
$fs = 0.1;
$fa = 6;

difference() {
	union() {
		translate([0, 0, 15])
			cylinder(r = 15, h = 30);
		cylinder(r = 15, h = 10);
		cylinder(r = 13, h = 20);
	}
	translate([-25, 15, 10])
		rotate([12, 0, 0])
		cube([ 50, 20, 60]);
	translate([-25, -32, 20])
		cube([ 50, 20, 60]);
	translate([0, 0, 42])
		cylinder(r = 2, h = 5);
}

