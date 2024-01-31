
h = 50;
b = 110;
t = 6;

steps = 4;
bsteps = 7;
factor = 1.3;

len = (h+2*t) * (steps + 1) - t;
scale = (len - 3*t) / (pow(factor, steps+1) - factor);

function fach_pos(i) = 2*t + scale * (pow(factor, i+1) - factor);

explode = 10;

module fach()
 {
	square([ b, h ]);
	translate([ -t, h-2*t ]) square([ t*2, t ]);
	translate([ -t, t ]) square([ t*2, t ]);
	translate([ b-t, h-2*t ]) square([ t*2, t ]);
	translate([ b-t, t ]) square([ t*2, t ]);
	translate([ t, -t ]) square([ t, t*2 ]);
	translate([ b-t*2, -t ]) square([ t, t*2 ]);
}

module fstack()
{
	for (i = [0 : steps])
		translate([ t, fach_pos(i), 2*t ]) rotate([ 90, 0, 0 ])
			linear_extrude(height = t) fach();
}

module bottom()
{
	projection(cut = true)
	translate([ 0, 0, -t*1.5 ])
	difference() {
		translate([ t, 0, t ])
		cube([ b, len, t ]);
		fstack();
	}

	for (i = [0 : bsteps]) {
		translate([ 0, t + i*(len-3*t) / bsteps, 0 ]) square([ t*1.5, t ]);
		translate([ b+t/2, t + i*(len-3*t) / bsteps, 0 ]) square([ t*1.5, t ]);
	}
}

module side()
{
	projection(cut = true)
	translate([ 0, 0, t/2 ])
	rotate([ 0, 90, 0 ])
	difference() {
		cube([ t, len, h+2*t ]);
		fstack();
		translate([ 0, 0, t ]) linear_extrude(height = t) bottom();
	}
}

module layout()
{
	side();
	translate([ h+t*3, 0 ]) bottom();
	translate([ h+b+t*6, len ]) mirror([ 0, 1 ]) side();
	for (i = [0 : steps])
		translate([ 2*h+b+t*10, (h+2*t)*i + t ]) fach();
}

module assembled()
{
	translate([ t-explode, 0, 0 ]) rotate([ 0, -90, 0 ])
	color([ 0.3, 0.7, 0.3 ]) linear_extrude(height = t) side();

	translate([ 0, 0, t ])
	color([ 0.7, 0.3, 0.3 ]) linear_extrude(height = t) bottom();

	translate([ b+2*t+explode, 0, 0 ]) rotate([ 0, -90, 0 ])
	color([ 0.3, 0.7, 0.3 ]) linear_extrude(height = t) side();

	translate([ 0, 0, explode ])
	color([ 0.3, 0.3, 0.7 ]) fstack();
}

// layout();
assembled();
