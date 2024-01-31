
use <rs60112.scad>;
use <lorlinck.scad>;
use <cnl.scad>;

module socket4mm(col)
{
	difference() {
		union() {
			color([ 0.7, 0.7, 0.75 ]) cylinder(r = 3, h = 15, center = true, $fn = 12);
			color(col) cylinder(r1 = 5, r2 = 3.5, h = 10, $fn = 12);
		}
		color(col) cylinder(r = 2, h = 12, $fn = 12);
	}
}

module frontpanel_2d()
{
	difference() {
		translate([ -70, -70 ]) difference() {
			square([ 170, 130 ]);
			for (y = [13/2:26*3:130]) {
				translate([ -4, y ]) square([ 8, 13*3 ]);
				translate([ 170-4, y ]) square([ 8, 13*3 ]);
			}
		}
		for (x = [ -50, -30, -10, +10, +30, +50 ]) translate([ x*7*2.54/20, 0 ]) {
			square([2, 67], center=true);
			for (y = [-1, +1]) translate([0, y*71/2]) circle(r = 1.8/2, $fn = 24);
			translate([0, -55 - 2.54]) square([2.54*3+2, 2.5], center=true);
			translate([0, -55 + 2.54]) square([2.54*3+2, 2.5], center=true);
		}
		for (x = [ -60, -20, +20, +60 ]) translate([ x*7*2.54/20, -55 ]) {
			circle(r = 1.8/2, $fn = 24);
		}
		for (x = [ 65, 85 ]) translate([ x, 25 ]) {
			circle(r = 3.5, $fn = 24);
			translate([ 0, 3.395 ]) rotate(45) square(1.1, center = true);
		}
		translate([ 75, -15 ]) {
			circle(r = 10/2, $fn = 24);
			for (i = [ -2.5 : 1 : +2.5 ])
				rotate(i*30) translate([ 0, +9.5 ]) circle(r = 3/2, $fn = 24);
		}
		translate([ 100-2, -70 + 13*5 ]) cnlA([ -1, 0 ]);
		translate([ -70+2, -70 + 13*5 ]) cnlA([ +1, 0 ]);
		translate([ 100-4, -70 + 13*2 ]) rotate(180) cnlB();
		translate([ -70+4, -70 + 13*2 ]) cnlB();
		translate([ 100-4, -70 + 13*8 ]) rotate(180) cnlB();
		translate([ -70+4, -70 + 13*8 ]) cnlB();

		translate([ 60, -40 ]) square([ 5, 4 ]);
		translate([ 85, -40 ]) square([ 5, 4 ]);
	}
}

module sidepanel_2d()
{
	translate([ -cos(30)*70, -sin(30)*70 ]) difference() {
		square([ cos(30)*130, sin(30)*130 ]);
		rotate(30) union() {
			difference() {
				square([ 130, 130 ]);
				for (y = [13/2:26*3:130])
					translate([ y, -4 ]) square([ 13*3, 8 ]);
			}
			translate([ 13*2, 2 ]) cnlA([ 0, -1 ]);
			translate([ 13*5, 0 ]) rotate(-90) cnlB();
			translate([ 13*8, 2 ]) cnlA([ 0, -1 ]);
		}
		translate([ 105.641, 20 ]) {
			translate([ 0, 10 ]) square([ 4, 10 ], center = true);
			translate([ 0, -10 ]) square([ 4, 10 ], center = true);
			cnlA();
		}
	}
}

module backpanel_2d()
{
	difference() {
		translate([ -70, -30 ]) square([ 170, 30]);
		translate([ -74, -20 ]) square([ 8, 10 ]);
		translate([ +96, -20 ]) square([ 8, 10 ]);
		translate([ +96, -15 ]) rotate(180) cnlB();
		translate([ -66, -15 ]) cnlB();
	}
}

module fader_spacer_2d()
{
	translate([ 0, +2.5/2 ]) difference() {
		square([ 8, 5 ], center = true);
		translate([ 0, -2.5/2 ]) square([ 2, 2.5 ], center = true);
	}
}

module cableclip_2d()
{
	difference() {
		translate([ -15, -4 ]) square([ 30, 19 ]);
		translate([ -8, -9 ]) square([ 16, 19 ]);
		translate([ -10, -19 ]) square([ 20, 19 ]);
	}
}

module assembled()
{
	rotate([30, 0, 0])
	union() {
		translate([ 0, 0, -5 ]) rs60112_bundle();
		translate([ 75, -15, 0 ]) lorlinck();
		
		translate([ 65, +25, 4 ]) socket4mm([ 0.3, 0.3, 0.3 ]);
		translate([ 85, +25, 4 ]) socket4mm([ 0.8, 0.3, 0.3 ]);
		
		linear_extrude(height = 4) frontpanel_2d();
		color([ 0.8, 0.2, 0.2 ]) for (x = [ -50, -30, -10, +10, +30, +50 ]) {
			translate([ x*7*2.54/20, +71/2, -4.5 ]) linear_extrude(height = 4) fader_spacer_2d();
			translate([ x*7*2.54/20, -71/2, -4.5 ]) linear_extrude(height = 4) rotate(180) fader_spacer_2d();
		}

		color([ 0.8, 0.2, 0.2 ]) translate([ 75, -38 ]) rotate([ -90, 0, 0 ])
			linear_extrude(height = 4, center = true) cableclip_2d();
	}

	for (x = [-70+2, 100-2]) 
		color([ 0.4, 0.5, 0.6 ]) translate([ x, 0, 0 ]) rotate([ 90, 0, 90 ])
			linear_extrude(height = 4, center = true) sidepanel_2d();

	color([ 0.4, 0.7, 0.3 ]) translate([ 0, 45, 0 ]) rotate([ 90, 0, 0 ])
		linear_extrude(height = 4, center = true) backpanel_2d();
}

module parts()
{
	border = 20;
	% translate([ -border, -border ]) difference() {
		square([ 297, 210 ]);
		translate([ border, border ]) square([ 297-2*border, 210-2*border ]);
	}

	translate([ 71, 71 ]) frontpanel_2d();
	translate([ 71, 165 ]) backpanel_2d();
	translate([ 175 , 0 ]) rotate(-90) translate([ -53, 35 ]) sidepanel_2d();
	translate([ 240, 20 ]) rotate(-60) translate([ -53, 35 ]) mirror() sidepanel_2d();
	for (x = [1 : 6], y = [1 : 2]) translate([ 175 + 10*x, 140 + 10*y ]) fader_spacer_2d();
	for (x = [ 1, 4, 5 ], y = [ 0 ]) translate([ 175 + 10*x, 140 + 10*y ]) fader_spacer_2d();
	translate([ 199, 125 ]) rotate(30) cableclip_2d();
}

// parts();
assembled();

