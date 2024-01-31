
module rs60112(faderpos = 0.5, knobcolor = [0.8, 0.2, 0.2])
{
	// http://www.alps.com/WebObjects/catalog.woa/E/HTML/Potentiometer/SlidePotentiometers/RS__1/RS60112A600N.html
	color([ 0.7, 0.7, 0.75 ]) difference() {
		translate([0, 0, -7/2]) cube([9, 75, 7], center = true);
		for (y = [+1, -1])
			translate([0, y*71/2, 0]) cylinder(r = 2.4/2, h = 20, center = true, $fn = 12);
		cube([2.5, 67, 5], center = true);
	}
	color([ 0.7, 0.7, 0.75 ]) translate([-1, -60/2 + 60*faderpos - 5/2, -5])
		cube([2, 5, 20]);
	color(knobcolor) translate([0, -60/2 + 60*faderpos, 15]) difference() {
		intersection() {
			cube([ 13, 25, 10 ], center = true);
			rotate([0, 90, 0]) cylinder(r = 12, h = 20, center = true);
			sphere(r = 13);
		}
		translate([0, 0, 20]) rotate([0, 90, 0]) cylinder(r = 18, h = 20, center = true);
	}
}

module rs60112_bundle()
{
	translate([-50*7*2.54/20, 0, 0])
		rs60112(0.0, [0.8, 0.2, 0.2]);
	translate([-30*7*2.54/20, 0, 0])
		rs60112(0.7, [0.8, 0.2, 0.2]);
	translate([-10*7*2.54/20, 0, 0])
		rs60112(0.2, [0.2, 0.3, 0.7]);
	translate([+10*7*2.54/20, 0, 0])
		rs60112(0.3, [0.2, 0.3, 0.7]);
	translate([+30*7*2.54/20, 0, 0])
		rs60112(1.0, [0.4, 0.4, 0.4]);
	translate([+50*7*2.54/20, 0, 0])
		rs60112(0.5, [0.4, 0.4, 0.4]);
}

rs60112_bundle();
