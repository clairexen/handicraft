
l = 28;    // outer length of box (mm)
t = 1.5;   // thickness of cover plate (mm)
d = 1.88;  // thickness of side walls (mm)
h = 10;    // outer height of box (mm)
k = 7;     // snap dimension (mm)
e = 0.5;   // spacibng between parts (mm)
v = 0.2;   // snap inversion (mm)
a = 60;    // snap angle (deg)

module part_top(s)
{
	vv = s > 0 ? v : 0;
	module side_profile() {
		len = s+d+k-vv;
		len_delta = cos(a)/sin(a) * (t+2*s);
		rotate([-90, 0, 0]) linear_extrude(height = d+k/2+2*s) translate([0,-t-2*s])
			polygon([[0, 0], [len+len_delta/2,0], [len-len_delta/2,t+2*s], [0,t+2*s]]);
	}
	translate([0, 0, h-t-s]) difference() {
		union() {
			translate([d-s, d-s, 0])
					cube([ l-d+2*s, l-2*d+2*s, t+2*s ]);
			for (pos = [-s, l-d-k/2-s])
				translate([-s, pos, 0]) side_profile();
			translate([vv, l/2+k/2+s, s])
				rotate([0, 90, -90]) linear_extrude(height = k+2*s)
					polygon([ [0,d], [k/2,0], [k/2,d], [0,k] ]);
			if (s > 0)
				translate([ -s, l/2-k/2-s, -k/2 ]) cube([ d+2*s, k+2*s, k/3 ]);
			translate([l-2*d-s, l/2+k/2, 0])
				rotate([-90, 0, -90]) linear_extrude(height = 2*d+2*s)
					polygon([ [-s,0], [ k/2, -t/2 ], [k+s,0], [k+d+s,k/2], [-d-s,k/2] ]);
		}
		for (i = [1:3])
			translate([ i*l/5, l/4, t/3 ])
				cube([ d, l/2, t ]);
	}
}

module part_bottom()
{
	difference() {
		cube([ l, l, h ]);
		translate([d, d, d]) cube([ l-2*d, l-2*d, h ]);
		translate([d+2, d+2, -d]) cube([ l-4*d, l-4*d, d*3 ]);
		part_top(e);
	}
}

module panel() {
	color([ 0.4, 0.8, 0.4 ]) part_bottom();
	rotate([ 180, 0, 0 ]) translate([0, d, -h])
		color([ 0, 0.8, 0.8 ]) part_top(0);
}

module show() {
	color([ 0.4, 0.8, 0.4 ]) part_bottom();
	color([ 0, 0.8, 0.8 ]) part_top(0);
}

module debug() {
	module cuthalf(k) {
		difference() {
			child(0);
			translate([-1, l/2+k, -1]) cube([ l+2, l/2+1, h+2 ]);
		}
	}
	color([ 0.4, 0.8, 0.4 ]) cuthalf(1) part_bottom();
	color([ 0, 0.8, 0.8 ]) cuthalf(0) part_top(0);
}

panel();


