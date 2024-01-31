
mil = 2.54 / 100;
t = 10 * mil;
s = 10 * mil;

w = 30;
h = 20;
opadsw = 5;
rounds = 8;

for (i = [0 : rounds*2 - 1]) {
	rotate(0) translate([ 0, h/2 - (t+s)*i ]) square([ w - 2*(t+s)*i + t, t ], center = true);
	rotate(-90) translate([ 0, w/2 - (t+s)*i ]) square([ h - 2*(t+s)*i + t, t ], center = true);
	rotate(-180) translate([ -t-s, h/2 - (t+s)*i ]) square([ w - 2*(t+s)*i + t - 2*(t+s), t ], center = true);
	rotate(-270) translate([ -t-s, w/2 - (t+s)*(i+2) ]) square([ h - 2*(t+s)*(i+1) + t, t ], center = true);
}

translate([ 0, h/2 - (t+s)*rounds*2 ]) square([ w - 2*(t+s)*rounds*2 + t, t ], center = true);
rotate(-90) translate([ 0, w/2 - (t+s)*rounds*2 ]) square([ h - 2*(t+s)*rounds*2 + t, t ], center = true);

translate([ (t+s)/2, -(t+s)/2 ]) difference() {
	square([ w - 2*(t+s)*rounds*2 + t - (t+s), h - 2*(t+s)*rounds*2 + t - (t+s) ], center = true);
	square([ s, h ], center = true);
}

difference() {
	translate([ -w/2 - opadsw - t/2 + t+t+s, -h/2 - t/2 ]) square([ opadsw, h+t ]);
	translate([ -w/2 - t/2 + t+s, -s/2 ]) mirror([ 1, 0 ]) square([ opadsw, s ]);
	translate([ -w/2 - t/2 + t, 0 ]) square([ s, h/2 - t/2 ]);
	translate([ -w/2 - t/2 + t, h/2 - t/2 - s ]) square([ s+t*2, s ]);
}
