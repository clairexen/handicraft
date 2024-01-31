
d1 = 27; // X-width
d2 =  4; // Half Y-width
d3 =  3; // Height above hole
d4 = 10; // Hole diameter
d5 =  5; // Height below hole
d6 =  3; // Pipe thickness
d7 =  2; // Extra plug height

$fs = 0.1;

module pipeplug_half()
{
	difference() {
		translate([ -d1/2, 0, -d4/2 -d5 ])
			cube([ +d1, +d2, +d3 +d4 +d5 +d7/2 ]);
		rotate([ 90, 0, 0 ])
			cylinder(h = 3*d2, r = d4/2, center = true);
	}
	translate([ -d1/2 -d6, 0, +d3 +d4/2 ])
		intersection() {
			cube([ +d1 +2*d6, +d2 +d6, d7 ]);
			union() {
				translate([ d7, 0, 0 ])
					cube([ +d1 +2*d6 -2*d7, +d2 +d6 -d7, d7 ]);
				for (x = [ d7, +d1 +2*d6 - d7 ])
					translate([ x, +d2 +d6 -d7, 0]) {
						rotate([ 90, 0, 0 ]) cylinder(r = d7, h = +d2 +d6 -d7);
						sphere(r = d7);
					}
				translate([ d7, +d2 +d6 -d7, 0 ]) rotate([ 0, 90, 0 ])
					cylinder(r = d7, h = d1 + 2*d6 -2*d7);
			}
		}
}

for (x = [ -d1/2 -d6 -1, +d1/2 +d6 +1 ], y = [ 0, +d3 +d4 +d5 +d7 +2 ])
	translate([ x, y, 0 ]) rotate([ 90, 0, 0 ]) pipeplug_half();
