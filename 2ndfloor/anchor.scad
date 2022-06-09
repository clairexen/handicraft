// Part List:
//
//     A1  A2   2x  113x40x40 slat (45/45)
//     B1       1x  153x40x40 slat ( 0/ 0)
//     C1  C2   2x  260x40x40 slat (45/ 0)
//     D1  D2   2x  487x40x40 slat (45/45)
//     E1       1x  509x40x40 slat ( 0/45)
//     F1  F2   2x  600x40x40 slat (45/45)
//     G1       1x  662x40x40 slat ( 0/45)
//     H1       1x  800x40x40 slat (45/45)
//     K1       1x  860x40x40 slat ( 0/ 0)
//     L1       1x  880x40x40 slat (45/45)
//     M1  M2   2x  889x40x40 slat (45/45)
//     N1       1x  900x40x40 slat ( 0/ 0)
//
//     P1  P2   2x   40x40x40 slat (45/ 0)
//     Q1       1x   40x40x40 slat (-45/-45)
//     R1       1x  120x40x40 slat (-45/-45)
//     S1       1x   80x40x40 slat (45/45/90)
//     T1       1x   80x40x40 slat (45/45/-90)
//
//  Total 23 slats with a combined length of 9862 mm.

explode = 0;

module slat(length=500, angle1=0, angle2=0, angle3=0, depth=40, height=40)
{
	ext1 = angle1 < 0 ? depth/cos(-angle1) : 0;
	ext2 = angle2 < 0 ? depth/cos(-angle2) : 0;
	echo(str(round(length), "x", height, "x", depth,
			" slat (", angle1, "/", angle2, "/", angle3, ")"));
	render() difference() {
		translate([-ext1, -depth, 0])
			cube([length+ext1+ext2, depth, height]);

		rotate([0, 0, angle1])
			translate([-depth, -depth*(0.5 + 1/cos(angle1)), -0.5*height])
			cube([depth, depth*(1 + 1/cos(angle1)), 2*height]);

		translate([length, 0, 0])
			translate([0, -depth/2, height/2])
			rotate([angle3, 0, 0])
			translate([0, depth/2, -height/2])
			rotate([0, 0, -angle2])
			translate([0, -depth*(0.5 + 1/cos(angle1)), -0.5*height])
			cube([depth, depth*(1 + 1/cos(angle1)), 2*height]);
	}
}

color([0.2,0.2,0.8])
	translate([-20-600*sin(45), 40+explode, -40-explode])
	slat(40+1200*sin(45), 45, 45);

color([0.2,0.8,0.2])
	translate([20, 40, 0])
	rotate([0, 0, -90])
	slat(900);

color([0.6,0.2,0.6])
	translate([20, 0, 40+explode])
	rotate([0, 0, -90])
	slat(900-40);

color([0.6,0.2,0.6])
	translate([20, -40-explode, -explode])
	rotate([90, 0, -90])
	slat(800*sin(45)-40/sin(45), 0, 45);

color([0.8,0.2,0.2])
	translate([-20, 0, -explode])
	rotate([0, 90, 0])
	slat(800*sin(45)+40/sin(45)+40, 0, 45);

color([0.2,0.2,0.8])
	translate([-20, 40+explode, 40/sin(45)-800*sin(45)-explode])
	rotate([0, 90, 0])
	slat(40 + 80/sin(45));

color([0.2,0.2,0.8])
	translate([-20, -40-2*explode, -800*sin(45)-explode])
	rotate([90, -135, 90])
	slat(800, 45, 45);

color([0.6,0.2,0.6])
	translate([-20, -40-3*explode, -40/sin(45)-800*sin(45)-explode])
	rotate([90, -135, 90])
	slat(800+80, 45, 45);

color([0.8,0.2,0.2])
	translate([20+600*sin(45)+explode, 40-explode, 0])
	rotate([0, 0, -135])
	slat(600, 45, 45);

color([0.8,0.2,0.2])
	translate([-20-explode, 40-600*sin(45)-explode, 0])
	rotate([0, 0, 135])
	slat(600, 45, 45);

color([0.2,0.2,0.8])
	translate([-20-600*sin(45), 40+explode, 40+explode])
	slat(40+1200*sin(45), 45, 45);

color([0.6,0.2,0.6])
	translate([60+40/sin(45)-600*sin(45), -40, 0])
	rotate([0, 0, 135])
	slat(80/sin(45), 45, 45);

color([0.6,0.2,0.6])
	translate([20-40/sin(45)+600*sin(45), 40, 0])
	rotate([0, 0, -135])
	slat(80/sin(45), 45, 45);

color([0.6,0.2,0.6])
	translate([20+explode, 40, 0])
	rotate([0, 0, -45])
	slat(300-40, 45, 0);

color([0.6,0.2,0.6])
	translate([-20-explode, 40,40])
	rotate([180, 0, -135])
	slat(300-40, 45, 0);

color([0.8,0.2,0.2])
	translate([20+explode, 40+explode, 0])
	slat(40, 45, 0);

color([0.8,0.2,0.2])
	translate([-60-explode, explode, 0])
	rotate([0, 0, 90])
	slat(40, 45, 0);

color([0.6,0.2,0.6])
	translate([-20+600*sin(45)+explode, 0, -2*explode])
	rotate([180, 0, 180])
	slat(80, 45, 45, 90);

color([0.6,0.2,0.6])
	translate([20-600*sin(45)-explode, 0, -40-2*explode])
	slat(80, 45, 45, -90);

k = 600 - 80/sin(45);

color([0.2,0.8,0.2])
	translate([-20-explode, 0, -k*sin(45)-explode])
	rotate([90, -135, 0])
	slat(k, 45, 45);

color([0.2,0.8,0.2])
	translate([20+k*sin(45)+explode, 0, -explode])
	rotate([90, 135, 0])
	slat(k, 45, 45);

color([0.2,0.2,0.8])
	translate([-20, explode, -k*sin(45)-explode])
	rotate([-90, 0, 0])
	slat(40, -45, -45);

color([0.6,0.2,0.6])
	translate([-60, explode, 40-k*sin(45)])
	rotate([-90, 0, 0])
	slat(120, -45, -45);
