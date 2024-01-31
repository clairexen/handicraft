
assembled();
// parts();

// ************************** parameters **************************

$fs = 0.3;
$fa = 5;

t = 6;
s = 3;
w = 200;
nrs = 10;

d1 = 350;
d2 = 330;
d3 = 200;
d4 = 50;
d5 = 25;
d6 = 50;

ar = 5;
ar_extra = 0.25;
mr = 6 / 2;
mf = 5 - mr;

cnl1 = 3;	// screw diameter
cnl2 = 5.5;	// nut diameter
cnl3 = 2.5;	// nut thickness
cnl4 = 18;	// max screw length
cnl5 = 12;	// min screw length

// ************************** captured nuts lock **************************

module cnlA() {
	circle(cnl1/2);
}

module cnlB() {
	render(convexity = 2) {
		translate([ -t, -cnl1/2 ]) square([ cnl4, cnl1 ]);
		translate([ cnl5-t-cnl3, -cnl2/2 ]) square([ cnl3, cnl2 ]);
	}
}

// ************************** gears **************************

module gear15OL() {
	difference() {
		import_dxf("gear15.dxf");
		difference() {
			circle(r = mr);
			translate([ -mr*2, mf ]) square(mr*4);
		}
	}
}

module gear50OL() {
	difference() {
		import_dxf("gear50.dxf");
		circle(r = ar);
		for (i = [1:nrs]) {
			rotate(360/nrs * i + 45) translate([ d5/4, d6 ]) square([ d5/4, t ]);
			rotate(360/nrs * i + 45) translate([ -d5/2, d6 ]) square([ d5/4, t ]);
			rotate(360/nrs * i + 45) translate([ 0, d6 + t/2 ]) cnlA();
		}
	}
}

module gear15() {
	color([ 0.8, 0.2, 0.2 ]) linear_extrude(height = t) gear15OL();
}

module gear50() {
	color([ 0.2, 0.8, 0.2 ]) linear_extrude(height = t) gear50OL();
}

module motorHolderOL() {
	translate([ 0, -7 ])
	difference() {
		circle(40/2);
		translate([ 0, 7 ]) circle(mr);
		for (phi = [ 30, -30, 90, -90, 180-30, 180+30 ])
			rotate(phi) translate([ 0, 31/2 ]) circle(3/2);
	}
}

module motorHolder() {
	color([ 0.2, 0.7, 0.7 ]) linear_extrude(height = t) motorHolderOL();
}

// ************************** frame **************************

module frameSideOL(type) {
	difference()
	{
		square(d1, center = true);
		circle(ar + ar_extra);
		if (type == "A") rotate(90+45) translate([ 50*5/2 + 15*5/2, 0, 0 ]) {
			circle(12/2);
			rotate(-45-90) translate([ 0, -7 ])
			for (phi = [ 30, -30, 90, -90, 180-30, 180+30 ])
				rotate(phi) translate([ 0, 31/2 ]) circle(3/2);
		}

		translate([ d2/4, d2/2 ]) square([ d2/4, t ]);
		translate([ -d2/2, d2/2 ]) square([ d2/4, t ]);
		translate([ -d2/5, d2/2 + t/2 ]) cnlA();
		translate([ +d2/5, d2/2 + t/2 ]) cnlA();

		rotate(90) translate([ d3/4-d3/4, d2/2-t ]) square([ d3/4, t ]);
		rotate(90) translate([ -d3/2-d3/4, d2/2-t ]) square([ d3/4, t ]);
		rotate(90) translate([ +d3/5-d3/4, d2/2-t/2 ]) cnlA();
		rotate(90) translate([ -d3/5-d3/4, d2/2-t/2 ]) cnlA();

		rotate(180+45) translate([ d4/4, d1/2 ]) square([ d4/4, t ]);
		rotate(180+45) translate([ -d4/2, d1/2 ]) square([ d4/4, t ]);
		rotate(180+45) translate([ 0, d1/2 + t/2 ]) cnlA();

		rotate(270+45) translate([ d4/4, d1/2 ]) square([ d4/4, t ]);
		rotate(270+45) translate([ -d4/2, d1/2 ]) square([ d4/4, t ]);
		rotate(270+45) translate([ 0, d1/2 + t/2 ]) cnlA();

		translate([ d2/2-t, d4/4 ]) square([ t, d4/4 ]);
		translate([ d2/2-t, -d4/2 ]) square([ t, d4/4 ]);
		translate([ d2/2 - t/2, 0 ]) cnlA();
	}
}

module frameSide(type) {
	linear_extrude(height = t) frameSideOL(type);
}

module frameTopOL() {
	difference() {
		translate([ -d2/2, -w/2 ])
			square([ d2, w ]);
		translate([ 0, -w/2 ]) square([ d2/2, t*2 ], center=true);
		translate([ 0, +w/2 ]) square([ d2/2, t*2 ], center=true);
		for (a = [-1, +1], b = [-1, +1])
			translate([ a*d2/5, b*(w-2*t)/2 ]) rotate(b*-90) cnlB();
		// FIXME: The d2/2 for the next cutout is not save for other d2 and w combinations!
		translate([ -100, 0 ]) scale([ 3.5, 1 ]) rotate(-45) square(d2/2);
	}
}

module frameTop() {
	linear_extrude(height = t) frameTopOL();
}

module frameFrontOL() {
	difference() {
		square([ d3, w ], center = true);
		translate([ 0, -w/2 ]) square([ d3/2, t*2 ], center=true);
		translate([ 0, +w/2 ]) square([ d3/2, t*2 ], center=true);
		for (a = [-1, +1], b = [-1, +1])
			translate([ a*d3/5, b*(w-2*t)/2 ]) rotate(b*-90) cnlB();
	}
}

module frameFront() {
	linear_extrude(height = t) frameFrontOL();
}

module frameBackOL() {
	difference() {
		square([ d4, w ], center = true);
		translate([ 0, -w/2 ]) square([ d4/2, t*2 ], center=true);
		translate([ 0, +w/2 ]) square([ d4/2, t*2 ], center=true);
		translate([ 0, -(w-2*t)/2 ]) rotate(90) cnlB();
		translate([ 0, (w-2*t)/2 ]) rotate(-90) cnlB();
	}
}

module frameBack() {
	linear_extrude(height = t) frameBackOL();
}


// ************************** wheel **************************

module wheelSideOL(type) {
	difference() {
		circle(155);
		circle(ar);
		for (i = [1:nrs]) {
			if (type == "A") {
				rotate(360/nrs * i) translate([ -d5/2, d6 ]) {
					translate([ d5, t ] / 2) circle(5);
					square([ d5, t ]);
				}				
			} else {
				rotate(360/nrs * i) translate([ d5/4, d6 ]) square([ d5/4, t ]);
				rotate(360/nrs * i) translate([ -d5/2, d6 ]) square([ d5/4, t ]);
				rotate(360/nrs * i) translate([ 0, d6 + t/2 ]) cnlA();
			}
		}
	}
}

module wheelSide(type) {
	linear_extrude(height = t) wheelSideOL(type);
}

module wheelSegmentOL() {
	translate([ 0, t/2-s/2 ]) difference() {
		square([ d5, w-4*t-s ], center = true);
		translate([ 0, -(w-4*t-s)/2 ]) square([ d5/2, t*2 ], center=true);
		translate([ 0, +(w-4*t-s)/2 ]) square([ d5/2, t*2 ], center=true);
		translate([ 0, -(w-6*t-s)/2 ]) rotate(90) cnlB();
		translate([ 0, (w-6*t-s)/2 ]) rotate(-90) cnlB();
	}
}

module wheelSegment() {
	linear_extrude(height = t) wheelSegmentOL();
}

module wheelSpacerOL() {
	difference() {
		circle(ar*3);
		circle(ar);
	}	
}

module wheelSpacer() {
	linear_extrude(height = t) wheelSpacerOL();
}

// ************************** assembled **************************

module assembled()
{
	translate([ 0, -w/2+t*2, 0 ]) rotate([ 90, -45-90, 0 ]) {
		translate([ 50*5/2 + 15*5/2, 0, -t ]) gear15();
		translate([ 0, 0, -s/2-t ]) gear50();
	}
	translate([ 0, -w/2+2*t, 0 ]) rotate([ 90, -45-90, 0 ])
	translate([ 50*5/2 + 15*5/2, 0, 0 ]) rotate([ 0, 0, -45-90 ]) motorHolder();
	
	%color([0.5, 0.5, 0.5]) translate([ 0, -w/2+t, 0 ]) rotate([ 90, 0, 0 ]) frameSide("A");
	%color([0.5, 0.5, 0.5]) translate([ 0, +w/2, 0 ]) rotate([ 90, 0, 0 ]) frameSide("B");
	
	translate([ 0, 0, d2/2 ]) frameTop();
	translate([ -d2/2, 0, -d3/4 ]) rotate([ 0, 90, 0 ]) frameFront();

	rotate([ 0, 45, 0 ]) translate([ d1*0.5, 0, 0 ]) rotate([ 0, 90, 0 ]) frameBack();
	rotate([ 0, -45, 0 ]) translate([ d1*0.5, 0, 0 ]) rotate([ 0, 90, 0 ]) frameBack();
	translate([ d2/2-t, 0, 0 ]) rotate([ 0, 90, 0 ]) frameBack();
	
	translate([ 0, -w/2+t*4+s/2, 0 ]) rotate([ 90, 0, 0 ]) wheelSide("A");
	translate([ 0, w/2-t*2-s/2, 0 ]) rotate([ 90, 0, 0 ]) wheelSide("B");
	
	color([ 0.2, 0.5, 0.5 ]) translate([ 0, -w/2+t*2+s/2, 0 ]) rotate([ 90, 0, 0 ]) wheelSpacer();
	color([ 0.2, 0.5, 0.5 ]) translate([ 0, w/2-t-s/2, 0 ]) rotate([ 90, 0, 0 ]) wheelSpacer();
	
	color([ 0.2, 0.5, 0.5 ]) translate([ 0, -w/2, 0 ]) rotate([ 90, 0, 0 ]) wheelSpacer();
	color([ 0.2, 0.5, 0.5 ]) translate([ 0, w/2+t, 0 ]) rotate([ 90, 0, 0 ]) wheelSpacer();
	
	for (i = [1:nrs]) {
		rotate([ 0, 360/nrs * i, 0 ])
		color([ 0.4, 0.4, 0.8 ]) translate([ 0, -s/2, d6 ]) wheelSegment();
	}

	color([ 0.6, 0.6, 0.6 ]) rotate([ 90, 0, 0 ])
		cylinder(r = ar, h = w+5*t, center = true);

	color([ 0.6, 0.6, 0.6 ]) rotate([ 0, 45, 0 ]) translate([ -50*5/2-15*5/2, -w/2, 0 ])
	rotate([ -90, -45, 0 ]) translate([ 0, 7, -50 ]) {
		cylinder(r = 40/2, h = 50);
		translate([ 0, -7, 0 ]) cylinder(r = 12/2, h = 56);
		translate([ 0, -7, 0 ]) difference() {
			cylinder(r = mr, h = 75);
			rotate(135) translate([ mf, -mr*1.5, 0 ]) cube([ mr*3, mr*3, 80 ]);
		}
	}
}


// ************************** parts **************************

module parts()
{
	module sheet(type) {
		w = 400;
		h = 400;
		render() difference() {
			square([ w, h ]);
			translate([ 10, 10 ]) square([ w-20, h-20 ]);
		}
	}

	for (x = [ 0, 500, 1000 ], y = [ 0, 500, 1000 ])
		color([ 0.8, 0, 0 ]) translate([ x, y ]) sheet(y == 0  ? "A" : "B");

	color([ 0.5, 0.5, 0.5 ])
	{
		translate([ d1/2 + 25, 0.5*d1 + 25 ]) frameSideOL("A");
		translate([ 500 + d1/2 + 25, 0.5*d1 + 25 ]) frameSideOL("B");
	
		translate([ 200, 700 ]) wheelSideOL("A");
		translate([ 700, 700 ]) wheelSideOL("B");

		translate([ 200, 1200 ]) {
			gear50OL();
			translate([ 130, 130 ]) gear15OL();
			translate([ -130, 130 ]) motorHolderOL();
			translate([ 120 + 20, -120 + 20  ]) wheelSpacerOL();
			translate([ 120 - 20, -120 - 20  ]) wheelSpacerOL();
			translate([ -120 + 20, -120 - 20 ]) wheelSpacerOL();
			translate([ -120 -20, -120 + 20 ]) wheelSpacerOL();
		}

		translate([ 1200, 700 ]) {
			translate([ 0, 0 ]) frameTopOL();
			translate([ 0, +150 ]) rotate(90) frameBackOL();
			translate([ 0, -150 ]) rotate(90) frameBackOL();
		}

		translate([ 1200, 200 ]) {
			translate([ -80, -80 ]) frameFrontOL();
			translate([ +60, -80 ]) frameBackOL();
			translate([ +105, 0 ]) for (i = [0:2]) {
				translate([ i*d5 + i*t, -90 ]) wheelSegmentOL();
				translate([ i*d5 + i*t, +90 ]) wheelSegmentOL();
			}
			translate([ -50, 50 ]) for (i = [0:3]) {
				translate([ 0, i*d5 + i*t ]) rotate(90) wheelSegmentOL();
			}
		}
	}
}
