
module lorlinck()
{
	// Lorlin CK series switch
	color([ 0.7, 0.7, 0.75 ]) {
		translate([0, 0, -2]) cylinder(r = 10/2, h = 10);
		translate([0, -9.5, -2]) cylinder(r = 3/2, h = 4, $fn = 12);
		translate([0, 0, -12.8]) cylinder(r = 22.2/2, h = 12.8);
	}
	color([ 0.3, 0.3, 0.3 ])
		cylinder(r = 6/2, h = 15); /* height of unmodified part: 50 */
	/* knob (just imaginary atm) */
	color([ 0.3, 0.8, 0.3 ]) translate([ 0, 0, 10 ])
		cylinder(r1 = 10, r2 = 8, h = 10);
	
}

lorlinck();

