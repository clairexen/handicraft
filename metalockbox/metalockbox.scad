h = 100;
w = 180;
l = 180;
t = 5;

p1y = 110;
p1x = 190;
p2x = 380;
p3x = 490;

module s1() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S1",
		height = t, origin = [   0,   0 ]); }
module s2() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S2",
		height = t, origin = [   0, p1y ]); }
module s3() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S3",
		height = t, origin = [ p2x, p1y ]); }
module s4() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S4",
		height = t, origin = [ p3x, p1y ]); }
module s5() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S5",
		height = t, origin = [ p1x, p1y ]); }
module s6() { dxf_linear_extrude(file = "metalockbox.dxf", layer = "S6",
		height = t, origin = [ p1x,   0 ]); }

module assembled() {
	translate([   0,   t,   0 ]) rotate([  90,   0,   0 ]) s1();
	translate([   0,   0,   0 ]) rotate([   0,   0,   0 ]) s2();
	translate([   t,   0,   0 ]) rotate([   0, -90,   0 ]) s3();
	translate([ t+w,   0,   0 ]) rotate([   0, -90,   0 ]) s4();
	%translate([   t,   0, h+t ]) rotate([   0,   -20,   0 ]) translate([ -t, 0, -t ]) s5();
	translate([   0, l+t,   0 ]) rotate([  90,   0,   0 ]) s6();
}

module exploded() {
	translate([   0,  -t,   0 ]) translate([   0,   t,   0 ]) rotate([  90,   0,   0 ]) s1();
	translate([   0,   0,  -t ]) translate([   0,   0,   0 ]) rotate([   0,   0,   0 ]) s2();
	translate([  -t,   0,   0 ]) translate([   t,   0,   0 ]) rotate([   0, -90,   0 ]) s3();
	translate([  +t,   0,   0 ]) translate([ t+w,   0,   0 ]) rotate([   0, -90,   0 ]) s4();
	%translate([   t,   0, h+t ]) rotate([   0,   -20,   0 ]) translate([ -t, 0, -t ]) s5();
	translate([   0,  +t,   0 ]) translate([   0, l+t,   0 ]) rotate([  90,   0,   0 ]) s6();
}

// exploded();
assembled();
