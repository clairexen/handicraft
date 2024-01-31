
module d30() {
	color([0.6,0.6,0.8]) rotate(-90,[0,0,1]) rotate(90,[1,0,0])
	linear_extrude(file="lazzzorrack.dxf", layer="d30", height = 50,
		origin = dxf_cross(file="lazzzorrack.dxf", layer="d30x"));
}

module s200() {
	color([0.6,0.8,0.6])
	linear_extrude(file="lazzzorrack.dxf", layer="s200", height = 50, origin = [ 0, -150 ]);
}

module s60() {
	color([0.8,0.6,0.6])
	linear_extrude(file="lazzzorrack.dxf", layer="s60", height = 50, origin = [ 1400, 0 ]);
}

module vprofile() {
	rotate(90,[1,0,0]) {
		linear_extrude(file="lazzzorrack.dxf", layer="vprofile1", height = 50);
		linear_extrude(file="lazzzorrack.dxf", layer="vprofile2", height = 50);
		linear_extrude(file="lazzzorrack.dxf", layer="vprofile3", height = 50);
	}

	% translate([ 0, -60, 0 ]) cube([ 600, 10, 600 ]);
}

module xprofile() {
	rotate(90,[1,0,0]) {
		linear_extrude(file="lazzzorrack.dxf", layer="xprofile1", height = 50, origin = [700, 0]);
		linear_extrude(file="lazzzorrack.dxf", layer="xprofile2", height = 50, origin = [700, 0]);
		linear_extrude(file="lazzzorrack.dxf", layer="xprofile3", height = 50, origin = [700, 0]);
	}
}

module rack()
{
	for (p = [0, 600, 900, 1200, 1500 ])
	translate([ 0, -p, 0 ]) vprofile();
	
	translate([ 0, -2000, 0 ]) mirror([0,1,0]) vprofile();
	
	for (p = [0, 100, 200, 300, 400, 500])
		translate([ 0, -1940, p]) %cube([ 600, 380, 10 ]);
	
	for (p = [600, 900, 1200, 1500])
		translate([ 600, -p, 0 ]) d30();
	
	translate([500, 0, 600]) rotate(-90) s200();
	
	translate([0, 0, -50]) rotate(-90) s200();
	translate([550/2, 0, -50]) rotate(-90) s200();
	translate([550, 0, -50]) rotate(-90) s200();
	
	translate([0, -100, -100]) s60();
	translate([0, -1075, -100]) s60();
	translate([0, -1950, -100]) s60();
	
	translate([0, 0, -400]) xprofile();
	translate([550, -50, -400]) rotate(180) d30();
	
	translate([0, -975, -400]) xprofile();
	translate([600, -975, -400]) d30();
	translate([550, -1025, -400]) rotate(180) d30();
	
	translate([0, -1950, -400]) xprofile();
	translate([600, -1950, -400]) d30();
	
	translate([0, 0, -450]) rotate(-90) s200();
	translate([550/2, 0, -450]) rotate(-90) s200();
	translate([550, 0, -450]) rotate(-90) s200();
	
	translate([0, -50, -500]) s60();
	translate([0, -1025, -500]) s60();
	translate([0, -2000, -500]) s60();
}

translate([-300, 1000, 0]) rack();
