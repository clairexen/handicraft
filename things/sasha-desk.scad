
H=1000;
D=1000;
W=1000;

module room() {
	color([0.7, 0.7, 0.6, 0.3]) render(convexity=5) difference() {
		translate(-[1000, 1000, 300]) cube([3000, 2500, 2000]);
		cube([6000, 2000, 6000]);
		translate([0,-500,0]) cube([1150, 2000, 3000]);
		translate([100,-2000,1000]) cube([950, 2500, 3000]);
	}
}

module table() {
	color([0.9, 0.9, 0.5]) intersection() {
		translate([0,-500,H]) cube([D,W, 3]);
		translate([0,500,-10]) rotate([0,0,270-atan2(W,D)]) cube([2000, 2000, 2000]);
	}
	translate([10,W-650,0]) cube([44,44,H]);
	translate([D-100,-490,0]) cube([44,44,H]);
	translate([10,-490,0]) cube([44,44,H]);
	color([0.9, 0.9, 0.5]) translate([0, -490, H/2]) cube([10, W-44-70, 200]);
	color([0.9, 0.9, 0.5]) translate([0, -490, H/2]) cube([D-44-10, 10, 200]);

}

%room();
table();
