
module dockA()
{
	difference() {
		union() {
			square([ 80, 150 ]);
			for (x=[-6,80-6], y=[10,100])
				translate([ x, y ])
					square([ 12, 10 ]);
		}
		translate([ 35, 50-6 ])
			difference() {
				square([ 45, 6 ]);
				translate([ 5, 0 ])
					square([ 35, 6 ]);
			}
	}
}

module dockB()
{
	d = sin(20)*150;
	o = sin(20)*6;
	difference() {
		polygon(points = [
			[ 0, 0 ],
			[ 0, -26 ],
			[ 50, -26 ],
			[ 50, -15 ],
			[ 150, -6 ],
			[ 150+cos(-20)*o, sin(-20)*o ],
			[ d*sin(20), d*cos(20) ]
		], paths = [
			[ 0, 1, 2, 3, 4, 5, 6, 7 ]
		]);
		translate([ 10, -6 ])
			square([ 10, 6 ]);
		translate([ 100, -6 ])
			square([ 10, 6 ]);
		translate([ 10, -26.5 ])
			square([ 30, 6.5 ]);
		rotate(-90-20)
			translate([ -d+6, 40 ])
				square([ 6, 50 ]);
	}
}

module dockC()
{
	difference() {
		union() {
			square([ 80, 50 ]);
			for (x=[-6,80-6])
				translate([ x, 10 ])
					square([ 12, 30 ]);
		}
		translate([ 35, 50-6 ])
			difference() {
				square([ 45, 6 ]);
				translate([ 5, 0 ])
					square([ 35, 6 ]);
			}
		translate([ 35+15, 40 ])
			circle(r = 2.5, $fs = 0.1);
		translate([ 80-15, 40 ])
			circle(r = 2.5, $fs = 0.1);
	}
}

module dockD()
{
	difference() {
		square([ 26, 45 ]);
		translate([ 0, 5 ])
			square([ 6, 35 ]);
		translate([ 20, 5 ])
			square([ 6, 35 ]);
	}
}

module dockE()
{
		difference() {
			union() {
				square([ 80, 70 ]);
				for (x=[-6,80-6])
					translate([ x, 10 ])
						square([ 12, 50 ]);
			}
			for (x=[20, 60], y=[20, 50])
				translate([ x, y ])
					circle(2);
		}
}

module dockF()
{
		difference() {
			translate([ 10, 10 ])
				square([ 60, 50 ]);
			for (x=[20, 60], y=[20, 50])
				translate([ x, y ])
					circle(2);
		}
}

module dock()
{
		d = sin(20)*150;
		color([ 0.8, 0.5, 0.2 ])
			linear_extrude(height = 6)
				dockA();
		color([ 0.8, 0.5, 0.2 ])
			translate([ 0, 0, 20 ])
				linear_extrude(height = 6)
					dockC();
		for (x=[0,80+6])
			translate([ x, 0, 0 ])
				rotate([ -90, 0, 90 ])
					linear_extrude(height = 6)
						dockB();
		color([ 0.6, 0.6, 0.2 ])
			translate([ 35, 50-6, 0])
				rotate([ 90, -90, 180 ])
					linear_extrude(height = 6)
						dockD();
		translate([ 0, sin(20)*(d-6), -cos(20)*(d-6) ])
			rotate([ 20, 0, 0 ])
				translate([ 0, 30, 0 ]) {
					color([ 0.8, 0.5, 0.2 ])
						linear_extrude(height = 6)
							dockE();
					color([ 0.6, 0.6, 0.2 ])
						translate([ 0, 0, -6 ])
							linear_extrude(height = 6)
								dockF();
				}
}

module cam()
{

	color([0.8, 0.2, 0.2])
		cube([61, 92, 14]);
	color([0.8, 0.8, 0.8])
		translate([ 4.5, -13, 4.5])
			cube([12.5, 15, 4.5]);
	color([0.2, 0.2, 0.2])
		translate([ (61-41)/2, 92-34-3.5, 13])
			cube([ 41, 34, 2 ]);
}

module led(col)
{
	$fs = 0.1;
	color(col) render() {
		translate([ 0, 0, 5 ])
			sphere(r=2.5);
		cylinder(r=2.5, h=5);
		cylinder(r=3, h=1);
	}
	color([0.8, 0.8, 0.8]) {
		translate([ -1, 0, -4 ])
			cylinder(r=0.5, h=5);
		translate([ +1, 0, -4 ])
			cylinder(r=0.5, h=5);
	}
}

module usb()
{
		difference() {
			union() {
				color([0.8, 0.8, 0.8])
					translate([ (19-15.5)/2, 1, (12-7)/2 ])
						cube([ 14.5, 43-1, 7 ]);
				color([0.8, 0.2, 0.3])
					cube([ 19, 38.5, 12 ]);
			}
			color([0.2, 0.2, 0.2])
				translate([ (19-12.5)/2, 2, (12-5)/2 ])
					cube([ 12.5, 43-1, 5 ]);
		}
}

module preview()
{
	translate([ 0, -sin(20)*150, 0 ])
		rotate([ 90-20, 0, 0 ])
			dock();
	
	translate([ 0, -sin(20)*150, 0 ])
		rotate([ 90-20, 0, 0 ]) {
			translate([ 35+15, 40, 19 ])
				led([ 1, 0.5, 0.5, 0.5 ]);
			translate([ 80-15, 40, 19 ])
				led([ 0.5, 1, 0.5, 0.5 ]);
		}
	
	translate([ 0, -sin(20)*150, 0 ])
		rotate([ 90-20, 0, 0 ])
			translate([ (80-61)/2, 52, 8 ])
				cam();
	
	translate([ 0, -sin(20)*150, 0 ])
		rotate([ 90-20, 0, 0 ])
			translate([ 11, 8, 8 ])
				usb();
}

module layout()
{
	translate([ 180, 10 ])
		dockA();
	translate([ 10, 60 ]) {
		rotate(20) dockB();
		translate([ 150, -20 ])
		rotate(10) mirror() dockB();
	}
	translate([ 20, 120 ])
		dockC();
	translate([ 120, 120 ])
		dockD();
	translate([ 20, 180 ])
		dockE();
	translate([ 120, 180 ])
		dockF();
}

preview();
// layout();
