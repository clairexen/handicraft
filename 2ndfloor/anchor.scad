
module slat(length=500, angle1=0, angle2=0, depth=40, height=40)
{
	ext1 = angle1 < 0 ? depth/cos(-angle1) : 0;
	ext2 = angle2 < 0 ? depth/cos(-angle2) : 0;
	render() difference() {
		translate([-ext1, -depth, 0])
			cube([length+ext1+ext2, depth, height]);

		rotate([0, 0, angle1])
			translate([-depth, -depth*(0.5 + 1/cos(angle1)), -0.5*height])
			cube([depth, depth*(1 + 1/cos(angle1)), 2*height]);

		translate([length, 0, 0]) rotate([0, 0, -angle2])
			translate([0, -depth*(0.5 + 1/cos(angle1)), -0.5*height])
			cube([depth, depth*(1 + 1/cos(angle1)), 2*height]);
	}
}

color([0.2,0.2,0.8])
	translate([-500, 40, -40])
	slat(1000);

color([0.2,0.8,0.2])
	translate([20, 40, 0])
	rotate([0, 0, -90])
	slat(900);

color([0.8,0.2,0.2])
	translate([-20, 0, 0])
	rotate([0, 90, 0])
	slat(800);

color([0.2,0.2,0.8])
	translate([-20, 40, -600])
	rotate([0, 90, 0])
	slat(200);

color([0.2,0.2,0.8])
	translate([-20, -40, -800*sin(45)])
	rotate([90, -135, 90])
	slat(800, 45, 45);

color([0.8,0.2,0.2])
	translate([20 + 600*sin(45), 40, 0])
	rotate([0, 0, -135])
	slat(600, 45, 45);

color([0.8,0.2,0.2])
	translate([-20, 40 - 600*sin(45), 0])
	rotate([0, 0, 135])
	slat(600, 45, 45);

k = 600 - 80/cos(45);

color([0.2,0.8,0.2])
	translate([-20, 0, -k*sin(45)])
	rotate([90, -135, 0])
	slat(k, 45, 45);

color([0.2,0.8,0.2])
	translate([20 + k*sin(45), 0, 0])
	rotate([90, 135, 0])
	slat(k, 45, 45);

color([0.2,0.2,0.8])
	translate([-20, 0, -k*sin(45)])
	rotate([-90, 0, 0])
	slat(40, -45, -45);
