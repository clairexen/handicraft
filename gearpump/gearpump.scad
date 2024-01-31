
mode =
	"2d";
//	"3d";

$fs = 0.2;

module g3(h = 10) difference() {
	linear_extrude(file = "gears.dxf", layer = "g3",
		height = h, convexity = 2);
	linear_extrude(file = "gears.dxf", layer = "lock",
		height = h*2+1, center = true, convexity = 2);
}

module h3(h = 10)
cylinder(r = 20.5, h = h);

module g15(h = 5) difference() {
	linear_extrude(file = "gears.dxf", layer = "g15",
		height = h, convexity = 2);
	linear_extrude(file = "gears.dxf", layer = "lock",
		height = h*2+1, center = true, convexity = 2);
}

module h15(h = 10)
cylinder(r = 16.5, h = h);

module ax(h = 5)
linear_extrude(file = "gears.dxf", layer = "axis",
		height = h, convexity = 2, $fn = 10);

module gear() {
	g15(5);
	translate([ 0, 0, 5 ]) g3(20);
	translate([ 0, 0, 25 ]) g15(5);
}

module hole() {
	translate([ 0, 0, -0.05 ]) {
		h15(5.1);
		translate([ 0, 0, 5 ]) h3(20.1);
		translate([ 0, 0, 25 ]) h15(5.1);
	}
}

module box1(extra_h = 0) {
	difference() {
		translate([ -25, -25, -5-extra_h ])
			cube([ 80, 50, 20 + extra_h ]);
		hole();
		translate([ 30, 0, 0 ]) hole();
		translate([ 15, 0, 15 ]) rotate(90, [1, 0, 0])
				cube([ 10, 10, 100], center = true);
		for (x = [ -20, +50], y = [ -20, +20 ])
			translate([ x, y ]) cylinder(r=2, h=50, center=true);
	}
}

module box2() {
	difference() {
		translate([ 0, 0, 30 ]) mirror([ 0, 0, 1 ]) box1(5);
		cylinder(r = 6, h = 40);
		translate([ 7, 0, 35 ]) cylinder(r = 38/2, h = 10);
	}
}

module pump(transparent_mode = false) {
	box1();
	if (transparent_mode)
		%box2();
	if (!transparent_mode)
		box2();
	rotate(120*$t) difference() {
		color([ 0.7, 0.3, 0.3 ]) gear();
		color([ 0.7, 0.7, 0.7 ]) ax(40);
		color([ 0.7, 0.7, 0.7 ]) translate([ 0, 0, 25.1 ]) cylinder(r = 6, h = 5);
	}
	color([ 0.3, 0.7, 0.3 ]) translate([ 30, 0, 0 ])
		rotate(-120*$t) gear();
}

module slice_planes() {
	for (i = [0:8]) {
		color([1,0,0 ]) translate([ +15, 0, -2.5 + i*5 ])
			cube([ 100, 70, 0.2 ], center = true);
	}
}

module slices(stepx = 90, stepy = -60) {
	tab = [ [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2] ]; 
	translate([ -15 - stepx, -stepy ]) for (i = [0:8]) {
		translate([ tab[i][0]*stepx, tab[i][1] * stepy ]) projection(cut = true)
			translate([ 0, 0, +2.5 - i*5 ]) pump();
	}
}

if (mode == "3d") {
	// slice_planes();
	pump(transparent_mode = true);
}

if (mode == "2d") {
	slices();
}
