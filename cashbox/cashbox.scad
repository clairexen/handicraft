
<cashbox_dxf.scad>

module box1() {
	A4();
	translate([0, 4, 100]) rotate(90, [1, 0, 0]) A2();
	translate([0, -100, 100]) rotate(90, [1, 0, 0]) B1();
	translate([0, 0, 96]) B2();
	translate([96, 0, 100]) rotate(90, [0, 1, 0]) A1();
}

module box2() {
	translate([-5, -90, 0]) rotate(90, [0, 0, 1]) B4();
	translate([0, 4, 100]) rotate(90, [1, 0, 0]) B3();
	translate([0, -90, 100]) rotate(90, [1, 0, 0]) A3();
	translate([0, -95, 100]) rotate(90, [0, 0, 1])
			rotate(90, [1, 0, 0]) A5();
	translate([91, -95, 100]) rotate(90, [0, 0, 1])
			rotate(90, [1, 0, 0]) B5();
	translate([4, -95, 86]) rotate(90, [0, 0, 1]) C1();
}

%translate([-50, 55, -55])
box1();

translate([-50, 50, -50])
box2();

