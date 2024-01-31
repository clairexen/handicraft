
mode =
//	"shapes";
	"3dview";

num_shelves = 4;
extra_h = 50;
round_r = 35;

shelf_w = 235;
shelf_h = (300-extra_h)/4;
shelf_d = 320;

thickness = 6;
locks_h = 10;
locks_w = 5;
locks_d = 5;

explode = (sin($t*360 - 90)+1)*10;
total_h = num_shelves*shelf_h + extra_h;

function shelf_hpos(i) = thickness + i*shelf_h;

module backside()
{
	difference() {
		union() {
			square([ shelf_w, total_h ]);
			for (i = [0:locks_h]) {
				translate([ 0, 2*thickness + (total_h-5*thickness)*i/locks_h ]) {
					translate([ -thickness, 0 ]) square([ thickness*2, thickness ]);
					translate([ shelf_w-thickness, 0 ]) square([ thickness*2, thickness ]);
				}
			}
		}
		for (i  = [0:num_shelves]) {
			for (j  = [0:locks_w])
				translate([ 2*thickness+(shelf_w-5*thickness)*j/locks_w, shelf_hpos(i) ])
					square([ thickness, thickness ]);
		}
	}
}

module shelf()
{
	square([ shelf_w, shelf_d ]);
	for (i = [0:locks_d]) {
		translate([ 0, 2*thickness + (shelf_d-5*thickness)*i/locks_d ]) {
			translate([ -thickness, 0 ]) square([ thickness*2, thickness ]);
			translate([ shelf_w-thickness, 0 ]) square([ thickness*2, thickness ]);
		}
	}
	for (j  = [0:locks_w])
		translate([ 2*thickness+(shelf_w-5*thickness)*j/locks_w, shelf_d ])
			square([ thickness, thickness ]);
}

module side()
{
	difference() {
		square([ total_h, shelf_d+thickness*2 ]);
		for (i  = [0:num_shelves]) {
			for (j  = [0:locks_d])
				translate([ shelf_hpos(i), 2*thickness+(shelf_d-5*thickness)*j/locks_d ])
					square([ thickness, thickness ]);
		}		
		for (i = [0:locks_h]) {
			translate([ 2*thickness + (total_h-5*thickness)*i/locks_h, shelf_d ]) 
				square([ thickness, thickness ]);
		}
		translate([ total_h-round_r, -round_r ])
		difference() {
			square(round_r*2);
			translate([ 0, round_r*2]) circle(round_r);
		}
	}
}

if (mode == "shapes") {
	translate([ -shelf_w -3*thickness, shelf_d+thickness*3 ]) backside();
	translate([ -shelf_w -3*thickness, 0 ]) shelf();
	side();
}

if (mode == "3dview") {
	color([ 0.3, 0.7, 0.3 ]) for (i = [0:num_shelves])
		translate([ 0, 0, shelf_hpos(i) ]) linear_extrude(height = thickness) shelf();
	color([ 0.7, 0.3, 0.7 ]) for (i = [0:1])
		translate([ i*(shelf_w+thickness) - explode + i*2*explode, 0, 0 ]) rotate([ 0, -90, 0 ])
			linear_extrude(height = thickness) side();
	color([ 0.7, 0.3, 0.3 ]) translate([ 0, shelf_d+thickness + explode, 0 ]) rotate([ 90, 0, 0 ])
			linear_extrude(height = thickness) backside();
}
