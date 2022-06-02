// Part List:   (total 14 slats)
//
//   2x   40 x 40 x 113  (45/45)
//   2x   40 x 40 x 487  (45/45)
//   2x   40 x 40 x 600  (45/45)
//   1x   40 x 40 x 800  (45/45)
//   2x   40 x 40 x 889  (45/45)
//
//   1x   40 x 40 x 137  (0/0)
//   1x   40 x 40 x 646  (0/0)
//   1x   40 x 40 x 900  (0/0)
//
//   1x   40 x 40 x 40   (-45/-45)
//   1x   40 x 40 x 120  (-45/-45)

module slat(length=500, angle1=0, angle2=0, depth=40, height=40)
{
	ext1 = angle1 < 0 ? depth/cos(-angle1) : 0;
	ext2 = angle2 < 0 ? depth/cos(-angle2) : 0;
	echo(str(depth, "x", height, "x", round(length), " slat (", angle1, "/", angle2, ")"));
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
	translate([-20-600*sin(45), 40, -40])
	slat(40+1200*sin(45), 45, 45);

color([0.2,0.8,0.2])
	translate([20, 40, 0])
	rotate([0, 0, -90])
	slat(900);

color([0.8,0.2,0.2])
	translate([-20, 0, 0])
	rotate([0, 90, 0])
	slat(80+800*sin(45));

color([0.2,0.2,0.8])
	translate([-20, 40, 40/sin(45)-800*sin(45)])
	rotate([0, 90, 0])
	slat(80 + 40/sin(45));

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

color([0.2,0.2,0.8])
	translate([-20-600*sin(45), 40, 40])
	slat(40+1200*sin(45), 45, 45);

color([0.6,0.2,0.6])
	translate([60+40/sin(45)-600*sin(45), -40, 0])
	rotate([0, 0, 135])
	slat(80/sin(45), 45, 45);

color([0.6,0.2,0.6])
	translate([20-40/sin(45)+600*sin(45), 40, 0])
	rotate([0, 0, -135])
	slat(80/sin(45), 45, 45);

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

color([0.6,0.2,0.6])
	translate([-60, 0, 40-k*sin(45)])
	rotate([-90, 0, 0])
	slat(120, -45, -45);
