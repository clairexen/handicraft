room_h=3000;
room_w=3500;
room_d=2500;
wall=100;

height=2000;
width=2600;
depth=1800;

xc=1300;
yc=500;

x1=500;
x2=1500;
y1=500;
y2=2500;

t1=80;
t2=20;
t3=40;

h=height-t1-t2;
w=width-t3;
d=depth-t3;

steps=10;
fans=25;
boards=10;

color([0.8, 0.8, 0.8]) render(convexity = 2) difference() {
	translate([-wall, -wall, -wall])
		cube([room_d, room_w, room_h] + [wall, wall, wall]);
	translate([-2*wall, width+t3, 0])
		cube([3*wall, width, height]);
	cube([room_d, room_w, room_h]);
}

module bolt() {
	color([0.8, 0.3, 0.4]) render(convexity = 2) {
		translate([0, 0, -wall+10]) cylinder(wall+5, 10, 10);
		translate([0, 0, 10]) cylinder(10, 20, 20);
	}
}

module ybar(t, l) {
	color([0.5, 0.3, 0.2]) cube([t, l, t]);
	color([0.7, 0.5, 0.4]) translate([-1, -1, -1]) cube([t+2, t/3, t+2]);
	color([0.7, 0.5, 0.4]) translate([-1, l-t/3+1, -1]) cube([t+2, t/3, t+2]);
}
module xbar(t, l) { translate([0, t, 0]) rotate([0, 0, -90]) ybar(t, l); }
module zbar(t, l) { translate([0, t, 0]) rotate([90, 0, 0]) ybar(t, l); }

translate([x1, t1, h-t1/2]) rotate([-90, 0, 0]) bolt();
translate([x2, t1, h-t1/2]) rotate([-90, 0, 0]) bolt();

translate([t1, y1, h+t1/2]) rotate([-90, 0, -90]) bolt();
translate([t1, y2, h+t1/2]) rotate([-90, 0, -90]) bolt();

translate([0, 0, h]) ybar(t1, w);
translate([0, 0, h-t1]) xbar(t1, d);

translate([d-t1, 0, h]) ybar(t1, w);
translate([0, w-t1, h-t1]) xbar(t1, d);

translate([xc-t1/2, 0, h]) ybar(t1, w);
translate([xc-t1/2, w-t1, 0]) zbar(t1, h-t1);
translate([d-t1, yc-t1/2, 0]) zbar(t1, h);
translate([d-t1, 0, 0]) zbar(t1, h-t1);

color([0.5, 0.3, 0.2])
for (i = [1:steps]) {
	translate([d-t1/2, t1/4, i*h/(steps+1)])
	rotate([-90, 0, 0])
	cylinder(yc, t1/3, t1/3);
}

color([0.7, 0.5, 0.6, 0.5])
render(convexity=fans)
for (i = [1:fans]) {
	translate([i*xc/(fans+1), w-t1/2, 0])
	cylinder(h-t1, t1/3, t1/3);
}

color([0.7, 0.5, 0.6, 0.5])
for (i = [0:boards-1]) {
	translate([0, i*(w+t2)/boards, h+t1])
	cube([d, w/boards-t2, t2]);
}

translate([xc, w/2, 1000]) difference() {
	union() {
		sphere(500);
		cylinder(h-1000, 5, 5);
	}
	translate([300,100,200]) sphere(400);
}

for (k = [150, 300]) {
	translate([0, w-t3, h+k]) xbar(t3, d);
	translate([d-t3, yc-t3, h+k]) ybar(t3, w-yc+t3);

	color([0.7, 0.5, 0.4])
	translate([d-t3-1, w-t3-1, h+k-1])
	render() intersection() {
		rotate([0, 0, -45]) translate([-2.5, 0, 0]) cube([5, sqrt(2*t3*t3), t3+2]);
		cube([100, 100, 100]);
	}
}


translate([0, w, h-t1]) zbar(t3, 300+t1+t3);
translate([t3, w, h-t1]) zbar(t3, 300+t1+t3);

translate([d, yc, h-t1]) zbar(t3, 300+t1+t3);
translate([d, yc-t3, h-t1]) zbar(t3, 300+t1+t3);

translate([d, w-t3, h-t1]) zbar(t3, 300+t1+t3);
translate([d-t3, w, h-t1]) zbar(t3, 300+t1+t3);
translate([d, w, h-t1]) zbar(t3, 300+t1+t3);

