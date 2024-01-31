
pi = 3.14;

module lemon_sqeezer()
{
	h = 55;
	r = 40;
	rt = 10;
	riffs = 10;
	ri = r - (r*pi / riffs);
	rk = r/5;

	module outer()
	{
		intersection() {
			union() {
				for (i=[1:riffs]) {
					rotate([0, 0, i*360/riffs])
					translate([ri*2, 0, -h])
					rotate([0, atan2(-ri, h), 0])
					cylinder(r1 = 2*r*pi / riffs, r2 = 0, h = pow(h*h*4 + ri*ri*4, 0.5));
				}
				translate([0, 0, -h])
				cylinder(r1 = r, r2 = 0, h = 2*h);
			}
			union() {
				cylinder(h = h-rt, r1 = r, r2 = rt/1.2);
				translate([0, 0, h-rt*1.5]) sphere(rt);
				intersection() {
					sphere(r);
					translate([-1.5*r, -1.5*r, -3*r])
					cube(3*r);
				}
			}
		}
	
		translate([0, 0, -1.7*r])
		cylinder(r = r/3, h = 2*r);
	
		translate([0, 0, -2.25*r])
		intersection() {
			sphere(r);
			translate([-1.5*r, -1.5*r, r/2])
			cube(3*r);
		}
	
		translate([0, 0, -1.9*r])
		cylinder(r = r*1, h=r/4);
	}

	module inner()
	{
		translate([0, 0, -1.9*r - 1])
		cylinder(r = rk, h = h + 1);

		translate([0, 0, -1.9*r - 1])
		cylinder(r1 = 3*rk, r2 = rk, h = r/3 + 1);	

		translate([0, 0, h -1.9*r - 1])
		cylinder(r1 = 2*rk, r2 = 0, h = r);	

		translate([0, 0, h -1.9*r - 1 + r/2])
		cylinder(r1 = 2*rk, r2 = 0, h = r);	
	}

	difference() { outer(); inner(); }
}

module cutview()
{
	difference() {
		lemon_sqeezer();
		translate([ 0, 25, 0 ])
		cube([100, 50, 200], center = true);
	}
}

// cutview();
lemon_sqeezer();
