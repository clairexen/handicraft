
dxf_file = "calccubes_6mm_raw.dxf";

h = 54;
w = 54;
l = 54;
t = 6;

eyes = true;
space = t/2;

module s1() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S1", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C1"));
	if (eyes) difference() {
		s();
		translate([ 0*t, 0*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}
module s2() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S2", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C2"));
	if (eyes) difference() {
		s();
		translate([ -2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 2*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}
module s3() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S3", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C3"));
	if (eyes) difference() {
		s();
		translate([ -2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ 0*t, 0*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 2*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}
module s4() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S4", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C4"));
	if (eyes) difference() {
		s();
		translate([ -2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ -2*t, 2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 2*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}
module s5() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S5", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C5"));
	if (eyes) difference() {
		s();
		translate([ -2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ -2*t, 2*t, t/2 ]) sphere(t/2);
		translate([ 0*t, 0*t, t/2 ]) sphere(t/2);
		translate([ 2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 2*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}
module s6() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S6", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C6"));
	if (eyes) difference() {
		s();
		translate([ -2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ -2*t, 0*t, t/2 ]) sphere(t/2);
		translate([ -2*t, 2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, -2*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 0*t, t/2 ]) sphere(t/2);
		translate([ 2*t, 2*t, t/2 ]) sphere(t/2);
	}
	if (!eyes) s();
}

module box() {
	translate([ 0*(w/2+space), 1*(l/2+space), 0*(h/2+space) ])
		rotate([ 270, 180, 0 ]) s1();
	translate([ 0*(w/2+space), 0*(l/2+space), -1*(h/2+space) ])
		rotate([ 0, 180, 0 ]) s2();
	translate([ 1*(w/2+space), 0*(l/2+space), 0*(h/2+space) ])
		rotate([ 180, 270, 0 ]) s3();
	translate([ -1*(w/2+space), 0*(l/2+space), 0*(h/2+space) ])
		rotate([ 0, 270, 0 ]) s4();
	translate([ 0*(w/2+space), 0*(l/2+space), 1*(h/2+space) ])
		rotate([ 0, 0, 0 ]) s5();
	translate([ 0*(w/2+space), -1*(l/2+space), 0*(h/2+space) ])
		rotate([ 90, 0, 0 ]) s6();
}

box();
