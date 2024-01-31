
$fs=0.1;

alpha=45;
beta=45;
gamma=135;

a = 20;
b = 20;
c = 20;
d = 20;

// a*cos(-alpha) + b*cos(beta-alpha) + c*cos(gamma+beta-alpha) = d
// a*sin(-alpha) + b*sin(beta-alpha) + c*sin(gamma+beta-alpha) = 0

rotate([90, 0, 0])
	cylinder(h = 20, r = 3, center=true);

translate([d, 0, 0]) rotate([90, 0, 0])
	cylinder(h = 20, r = 3, center=true);

rotate([0, alpha, 0]) {
	translate([45, 0, 0])
		cube([100, 2, 10], center=true);
	translate([a, 0, 0]) {
		rotate([90, 0, 0])
			cylinder(h = 20, r = 3, center=true);
		rotate([0, -beta, 0]) {
			translate([b/2, 5, 0])
				cube([b + 10, 2, 10], center=true);
			translate([b, 0, 0]) {
				rotate([90, 0, 0])
					cylinder(h = 20, r = 3, center=true);
				rotate([0, -gamma, 0])
					translate([c/2, -5, 0])
						cube([c + 10, 2, 10], center=true);
			}
		}
	}
}

