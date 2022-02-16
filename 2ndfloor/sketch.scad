h=2000;
w=3000;
d=2000;
wall=100;

room_h=3000;
room_w=3500;
room_d=2500;

xc=1300;
yc=500;

x1=500;
x2=1500;
y1=500;
y2=2500;

t1=80;
t2=10;
t3=40;

steps=10;
fans=10;
boards=10;

color([0.8, 0.8, 0.8]) render(convexity = 2) difference() {
	translate([-wall, -wall, -wall])
	cube([room_d, room_w, room_h] + [wall, wall, wall]);
	cube([room_d, room_w, room_h]);
}

module bolt() {
	color([0.8, 0.3, 0.4]) render(convexity = 2) {
		translate([0, 0, -wall+10]) cylinder(wall+5, 10, 10);
		translate([0, 0, 10]) cylinder(10, 20, 20);
	}
}

module ybar(t, l) { color([0.5, 0.3, 0.2]) cube([t, l, t]); }
module xbar(t, l) { color([0.5, 0.3, 0.2]) cube([l, t, t]); }
module zbar(t, l) { color([0.5, 0.3, 0.2]) cube([t, t, l]); }

translate([x1, t1, h]) rotate([-90, 0, 0]) bolt();
translate([x2, t1, h]) rotate([-90, 0, 0]) bolt();

translate([t1, y1, h]) rotate([-90, 0, -90]) bolt();
translate([t1, y2, h]) rotate([-90, 0, -90]) bolt();

translate([0, 0, h-t1/2]) ybar(t1, w);
translate([0, 0, h-t1/2]) xbar(t1, d);

translate([d-t1, 0, h-t1/2]) ybar(t1, w);
translate([0, w-t1, h-t1/2]) xbar(t1, d);

translate([xc-t1/2, 0, h-t1/2]) ybar(t1, w);
translate([xc-t1/2, w-t1, 0]) zbar(t1, h);
translate([d-t1, yc-t1/2, 0]) zbar(t1, h);
translate([d-t1, 0, 0]) zbar(t1, h);

color([0.5, 0.3, 0.2])
for (i = [1:steps]) {
	translate([d-t1/2, t1/4, i*h/(steps+1)])
	rotate([-90, 0, 0])
	cylinder(yc, t1/3, t1/3);
}

color([0.7, 0.5, 0.6, 0.5])
for (i = [1:fans]) {
	translate([i*xc/(fans+1), w-t1/2, 0])
	cylinder(h, t1/3, t1/3);
}

color([0.7, 0.5, 0.6, 0.5])
for (i = [0:boards-1]) {
	translate([i*(d+t2)/boards, 0, h+t1/2])
	cube([d/boards-t2, w, t2]);
}

translate([xc, w/2, 1000]) difference() {
	union() {
		sphere(500);
		cylinder(h-1000, 5, 5);
	}
	translate([300,100,200]) sphere(400);
}

translate([0, w-t3, h+300]) xbar(t3, d);
translate([d-t3, yc, h+300]) ybar(t3, w-yc);
translate([0, w-t3, h]) zbar(t3, 300);
translate([d-t3, yc, h]) zbar(t3, 300);
translate([d-t3, w-t3, h]) zbar(t3, 300);
