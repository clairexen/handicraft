
// yellow thing dimensions
dia1 = 36.0;
dia2 = 40.5;
dia3 = 15.0;
num1 = 10;
h1 = 7;

// extra dimensions
thick1 = 2;
thick2 = 3.5;
dist1 = 30;
dist2 = 70;
dia4 = 8;

module profile(h = 20)
{
	translate([-thick2, 0, 0]) cube([2*thick2, thick2, h]);
	translate([-thick2/2, -thick2, 0]) cube([thick2, 2*thick2, h]);
}

module part1()
{
	$fn = 120;
	difference() {
		union() {
			cylinder(r = dia2/2 + thick1, h=h1);
			translate([0, -h1/2, 0])
				cube([dia2/2 + dist1, h1, h1 ]);
			translate([dia2/2 + dist1, 0, 0])
				cylinder(r = 2*thick2, h=h1);
		}
		translate([0, 0, -h1/2]) difference() {
			cylinder(r = dia2/2, h=2*h1);
			for (i = [1:num1-1])
				rotate([0, 0, 360/num1 * i + 180])
				translate([dia1/2+dia3/2, 0, 0])
				cylinder(r = dia3/2, h=2*h1);
		}
		translate([dia2/2 + dist1 + thick2*0.2, 0, -h1/2])
			rotate([0, 0, 90]) profile(2*h1);
		translate([-dia2/2-2*thick1, -thick1/2, -thick2/2])
			cube([dia2/2, thick1, 3*thick2]);
	}
}

module part2()
{
	$fn = 120;
	difference() {
		union() {
			translate([0, 0, thick2]) rotate([90, 180, 90]) profile(dist2);
			translate([dist2, 0, 0]) cylinder(r = dia4/2 + thick1, h = 2*thick2);
		}
		translate([dist2, 0, -thick2/2]) cylinder(r = dia4/2, h = 3*thick2);
		translate([dist2, -thick1/2, -thick2/2]) cube([dia4, thick1, 3*thick2]);
	}
}

part1();

translate([-dia2/2, dia2, 0])
part2();


