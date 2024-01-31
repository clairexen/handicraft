
// Generated using Clifford Wolf's boxgen script
// http://svn.clifford.at/handicraft/2009/boxgen

dxf_file = "example.dxf";

h = 120;
w = 234;
l = 124;
t = 4;

eyes = false;
space = 0; //t*2;
dspace = 0; //t*5;

module s1() {
	module s()
		dxf_linear_extrude(file = dxf_file, layer = "S1", center = true, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C1"));
	if (eyes) difference() {
		s();
		dxf_linear_extrude(file = dxf_file, layer = "M1", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C1"));
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
		dxf_linear_extrude(file = dxf_file, layer = "M2", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C2"));
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
		dxf_linear_extrude(file = dxf_file, layer = "M3", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C3"));
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
		dxf_linear_extrude(file = dxf_file, layer = "M4", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C4"));
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
		dxf_linear_extrude(file = dxf_file, layer = "M5", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C5"));
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
		dxf_linear_extrude(file = dxf_file, layer = "M6", center = false, convexity = 3,
			height = t, origin = dxf_cross(file = dxf_file, layer = "C6"));
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
	color([1,0,0]) translate([ 0*(w/2+space), 1*(l/2+space), 0*(h/2+space) ])
		rotate([ 270, 180, 0 ]) s1();
	color([0,1,0]) translate([ 0*(w/2+space), 0*(l/2+space), -1*(h/2+space) ])
		rotate([ 0, 180, 0 ]) s2();
	color([0,0,1]) translate([ 1*(w/2+space), 0*(l/2+space), 0*(h/2+space) ])
		rotate([ 180, 270, 0 ]) s3();
	color([1,1,0]) translate([ -1*(w/2+space), 0*(l/2+space), 0*(h/2+space) ])
		rotate([ 0, 270, 0 ]) s4();
	// translate([ 0*(w/2+space), 0*(l/2+space), 1*(h/2+space) + dspace ])
	//	rotate([ 0, 0, 0 ]) s5();
	color([1,0,1]) translate([ 0*(w/2+space), -1*(l/2+space), 0*(h/2+space) ])
		rotate([ 90, 0, 0 ]) s6();
}

box();
