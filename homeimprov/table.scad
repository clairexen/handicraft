
W=44;
gap=150;
ang=15;
width=800;
depth=700;
height=600;
strut=300;

module Part(length, angle1=0, angle2=0, angle3=0) {
	echo(str("PART L=", length, " A=(", angle1, ", ", angle2, ", ", angle3, ")"));
	render() difference() {
		cube([length, W, W]);
		rotate([0, angle1, 0])
			translate([-2*W, -W/2, -W/2])
			cube([2*W, 2*W, W+W/cos(angle1)]);
		translate([length, W/2, W/2])
			rotate([angle3, 0, 0])
			translate([0, -W/2, -W/2])
			rotate([0, -angle2, 0])
			translate([0, -W/2, -W/2])
			cube([2*W, 2*W, W+W/cos(angle2)]);
	}
}


module BottomFrame() {
	translate([-width/2, -depth/2, W]) {
		color([1.0, 0.3, 0.4]) {
			Part(width);
			translate([0, depth-W, 0]) Part(width);
		}
		color([0.3, 1.0, 0.4]) {
			translate([W, 0, W]) Part(width-2*W);
			translate([W, depth-W, W]) Part(width-2*W);
		}
		color([0.3, 0.4, 1.0]) {
			translate([W, 0, W]) rotate([0, 0, 90]) Part(depth);
			translate([width, 0, W]) rotate([0, 0, 90]) Part(depth);
		}
		color([0.3, 1.0, 0.4]) {
			translate([W, W, 0]) rotate([0, 0, 90]) Part(depth-2*W);
			translate([width, W, 0]) rotate([0, 0, 90]) Part(depth-2*W);
		}
	}
}

module Leg() {
	color([1.0, 0.3, 0.4])
		translate([-width/2+W, gap/2, 3*W])
		rotate([0, -90, 0])
		Part(height-3*W);
	
	color([0.3, 0.8, 0.8])
		translate([-width/2, W/cos(ang) + gap/2 + height*tan(ang) - W*(1-cos(ang)), 0])
		rotate([ang, 0, 0])
		rotate([0, -90, 90])
		Part((height+W*sin(ang))/cos(ang), ang, ang, 180);
	
}

module Legs() {
	Leg();
	mirror([1, 0, 0]) Leg();
	mirror([0, 1, 0]) Leg();
	rotate([0, 0, 180]) Leg();
}

module TopFrame_Variant1() {
	color([0.3, 1.0, 0.4])
		translate([-width/2+W, gap/2, height-W])
		Part(width-2*W);
	module S() {
		color([0.3, 0.4, 1.0])
			translate([-width/2+W, gap/2, height - strut*sin(45)-W])
			rotate([0, -45, 0])
			Part(strut, 45, 45);
	}
	S();
	mirror([1, 0, 0]) S();

	color([0.3, 0.4, 1.0])
		let(len = gap - 2*W*(1-cos(ang)) + 2*W*tan(ang))
		translate([-width/2, -len/2, height-W])
		rotate([0, 0, 90])
		Part(len, ang, ang);

	color([0.3, 1.0, 0.4])
		translate([-width/2+W, -gap/2, height-W])
		rotate([0, 0, 90])
		Part(gap);
}

module TopFrameHalve() {
	color([0.3, 1.0, 0.4])
		translate([-width/2+W, gap/2, height-W])
		Part(width-2*W);
	module S() {
		color([0.3, 0.4, 1.0])
			translate([-width/2+W, gap/2, height - strut*sin(45)-W])
			rotate([0, -45, 0])
			Part(strut, 45, 45);
	}
	S();
	mirror([1, 0, 0]) S();

	color([0.3, 0.4, 1.0])
		let(len = gap - 2*W*(1-cos(ang)) + 2*W*tan(ang))
		translate([-width/2, -len/2, height-W])
		rotate([0, 0, 90])
		Part(len, ang, ang);

	color([0.3, 1.0, 0.4])
		translate([-width/2+W, -gap/2, height-W])
		rotate([0, 0, 90])
		Part(gap);
}

module TopFrame() {
	TopFrameHalve();
	rotate([0, 0, 180]) TopFrameHalve();
}

module TableTop() {
	for (a = [0, 180]) rotate([0, 0, a]) {
		color([0.3, 1.0, 0.4])
			translate([-width/2 - 2*W, depth/2, height - W])
			rotate([0, 0, -90])
			Part(depth);

		color([0.3, 0.4, 1.0])
			translate([-width/2 + W, gap/2-W, height - W])
			Part(width - 2*W);
	}

	for (i = [-2, -1, 0, 1, 2]) {
		color([0.5, 0.6, 0.8])
			translate([-width/2 - 2*W, -depth/10 + i*(depth/5) + 2, height])
			cube([width + 4*W, depth/5 - 4, 10]);
	}
}

BottomFrame();
Legs();
TopFrame();

translate([0, 0, 4*W]) TableTop();
