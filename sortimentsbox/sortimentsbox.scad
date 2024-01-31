

$fs = 0.3;
$fa = 5;

$explode = 20;


// ************************** captured nuts lock **************************

cnl1 = 3;	// screw diameter
cnl2 = 5.5;	// nut diameter
cnl3 = 2.5;	// nut thickness
cnl4 = 17;	// max screw length
cnl5 = 15;	// min screw length
cnl6 = 10;	// hole diameter for screw head

module cnlA() {
	circle(cnl1/2);
}

module cnlB(t) {
	render(convexity = 2) {
		translate([ -t, -cnl1/2 ]) square([ cnl4, cnl1 ]);
		translate([ cnl5-t-cnl3, -cnl2/2 ]) square([ cnl3, cnl2 ]);
	}
}

module cnlC() {
	square(cnl6, center = true);
}


// ************************** box **************************

module box(
		mode = "assembled",
		pp = [ 50, 100, 150 ],
		t1 = 2,
		t2 = 6,
		sk = 5,
		fs = 20,
		h = 80,
		w = 100,
		d = 200,
		l = 25,
		s = 20)
{
	col = [
		[ 0.8, 0.2, 0.2 ],
		[ 0.2, 0.8, 0.2 ],
		[ 0.2, 0.2, 0.8 ],
	];

	fh = sqrt(pow(h-2*t2, 2) + pow(s-t2, 2));

	module ext1()
	{
		linear_extrude(height = t1)
			for (i = [ 0:$children-1 ])
				child(i);
	}

	module ext2()
	{
		linear_extrude(height = t2)
			for (i = [ 0:$children-1 ])
				child(i);
	}

	module floor()
	{
		difference() {
			square([ d-s, w ]);
			for (x = [0:1], y = [0:1])
				translate([ x*3*(d-s)/4, y*(w-t2) ])
				square([ (d-s)/4 , t2 ]);
			translate([ 0, w/3 ])
				square([ t2, w/3 ]);
			for (p = pp)
				translate([ p - t1/2, w/3 ])
					square([ t1, w/3 ]);
		}
	}

	module side()
	{
		alpha = atan2(h, s - t1);
		beta = atan2(h/2 - l/2, s-2*t2+fs);
		gamma = alpha - beta;
		x = sqrt(pow(h/2 - l/2, 2) + pow(s-2*t2+fs, 2));
		y = x * cos(gamma);
		z = x * sin(gamma);
		difference() {
			union() {
				square([ d, h ]);
				square([ d+fs, h/2 + l/2 ]);
			}
			translate([ (d-s)/4, 0 ]) square([ (d-s)/2, t2 ]);
			translate([ 0, h/3 ]) square([ t2, h/3 ]);
			for (p = pp)
				translate([ p - t1/2, h/3 ]) square([ t1, h/3 ]);
			translate([ d-s, t2 ]) rotate(-atan2(s-t1, h-t2)) {
				translate([ -t1/2, 0 ]) square([ t1, fh/3 ]);
				translate([ -t1/2, 2*fh/3 ]) square([ t1, fh/3 ]);
			}
			translate([ d-t2+fs, (h-l)/2 + (l-t2) ]) square([ t2, t2 ]);
			translate([ d-t2+fs, (h-l)/2 - h ]) square([ t2, t2+h ]);
			translate([ d-s+t2, 0 ]) rotate(alpha) scale([ y, z ] / h)
				translate([ 0, -h ]) circle(r=h);
		}
	}

	module back()
	{
		difference() {
			square([ w, h ]);
			for (x = [0:1], y = [0:1])
				translate([ x*(w-t2), y*2*h/3 ])
				square([ t2, h/3 ]);
			for (x = [0:1])
				translate([ x*2*w/3, 0 ])
				square([ w/3, t2 ]);
		}
	}

	module partition()
	{
		back();
	}

	module front()
	{
		difference() {
			square([ w, fh ]);
			for (x = [0:1])
				translate([ x*(w-t2), fh/3 ])
				square([ t2, fh/3 ]);
		}
	}

	module handle()
	{
		difference() {
			square([ w, l ]);
			for (x = [0:1])
				translate([ x*(w-t2), t2 ])
				square([ t2, l-2*t2 ]);
		}
	}

	if (mode == "assembled") translate(-[ d, w, h ] / 2) {
		color(col[0]) translate([ 0, 0, -$explode ]) ext2() floor();
		color(col[1]) translate([ 0, t2 - $explode, 0 ])
			rotate([ 90, 0, 0 ]) ext2() side();
		color(col[1]) translate([ 0, w + $explode, 0 ])
			rotate([ 90, 0, 0 ]) ext2() side();
		color(col[2]) translate([ -$explode, 0, 0 ])
			rotate([ 90, 0, 90 ]) ext2() back();
		for (p = pp)
			% color(col[2]) translate([ p - t1/2, 0, 0 ])
				rotate([ 90, 0, 90 ]) ext1() partition();
		% color(col[2]) translate([ d-s, 0, t2 ])
			rotate([ 90+atan2(s-t1, h-t2), 0, 90 ])
			translate([ 0, 0, -t1/2 ]) ext1() front();
		color(col[2]) translate([ d-t2+fs+$explode, 0, (h-l)/2 ])
			rotate([ 90, 0, 90 ]) ext2() handle();
	}
	
	if (mode == "parts") {
		side();
		translate([ 0, 2*h+sk ]) mirror([ 0, 1 ]) side();
		translate([ 0, 2*h+2*sk ]) floor();
		translate([ d-s+sk+l, 2*h+2*sk ]) rotate(90) handle();
		translate([ h, 2*h+w+3*sk ]) rotate(90) back();
		translate([ h+fh+sk, 2*h+w+3*sk ]) rotate(90) front();
	}
}


// ************************** rack **************************

module rack(
		mode = "assembled",
		type = "A",
		l1 = 5,
		l2 = 7,
		t = 8,
		w = 512,
		d = 200,
		h = 82)
{
	dx = d + t;
	wx = w + 2*t;
	hx = h + 2*t;

	col = [
		type == "A" ? [ 0.8, 0.2, 0.2 ] : [ 0.6, 0.8, 0.3 ],
		type == "A" ? [ 0.2, 0.8, 0.2 ] : [ 0.4, 0.3, 0.4 ],
		type == "A" ? [ 0.2, 0.2, 0.8 ] : [ 0.3, 0.4, 0.4 ],
	];

	module ext()
	{
		linear_extrude(height = t)
			for (i = [ 0:$children-1 ])
				child(i);
	}

	module floor()
	{
		difference() {
			square([ wx, dx ]);

			for (x=[0:1], y=[0:2:l1-1])
				translate([ x*(wx-t), y*(dx/l1) ]) {
					square([ t, dx/l1 ]);
					translate([ t*(1-x), 0.5*(dx/l1) ])
						rotate(180*x) cnlB(t);
				}
			for (x=[0:1], y=[1:2:l1-1])
				translate([ x*(wx-t) + t/2, (y+0.5)*(dx/l1) ]) cnlA();

			for (x=[1:2:l2-1])
				translate([ x*(wx/l2), dx-t ]) {
					square([ wx/l2, t ]);
					translate([ (wx/l2)*0.5, 0 ]) rotate(-90) cnlB(t);
				}
			for (x=[0:2:l2-1])
				translate([ (x+0.5)*(wx/l2), dx-t/2 ]) cnlA();
		}
	}

	module side()
	{
		difference() {
			square([ dx, hx ]);
			for (x=[1:2:l1], y=[0:1])
				translate([ x*(dx/l1), y*(hx-t) ]) {
					square([ dx/l1, t ]);
					translate([ 0.5*dx/l1, t*(1-y) ])
						if (type == "A" || y > 0)
							rotate(90+y*180) cnlB(t);
						else
							cnlC();
				}
			for (x=[0:2:l1], y=[0:1]) {
				translate([ (x+0.5)*dx/l1, y*(hx-t) + t/2 ]) cnlA();
			}
			translate([ dx - t/2, hx/2 ]) cnlA();
			if (type != "A") {
				square([ dx/l1, t ]);
				translate([ (l1-1)*dx/l1, 0 ]) square([ dx/l1, t ]);
			}
			translate([ dx/l1, hx-t ]) square([ (l1-2)*dx/l1, t ]);
		}
	}

	module back()
	{
		difference() {
			translate([ t, 0 ])
				square([ wx - 2*t, hx ]);
			for (x=[0:2:l2], y=[0:1])
				translate([ x*(wx/l2), y*(hx-t) ]) {
					square([ wx/l2, t ]);
					translate([ 0.5*wx/l2, t*(1-y) ])
						if (type == "A" || y > 0)
							rotate(90+y*180) cnlB(t);
						else
							cnlC();
				}
			for (x=[1:2:l2], y=[0:1]) {
				translate([ (x+0.5)*wx/l2, y*(hx-t) + t/2 ]) cnlA();
			}
			translate([ t, hx/2 ]) cnlB(t);
			translate([ wx-t, hx/2 ]) rotate(180) cnlB(t);
			if (type != "A")
				square([ wx, t ]);
		}
	}

	if (mode == "assembled") rotate(90) translate(-[ wx, dx-t, hx ] / 2) {
		if (type == "A")
			color(col[0]) translate([ 0, 0, -$explode ]) ext() floor();
		color(col[0]) translate([ 0, 0, hx - t + $explode ]) ext() floor();
		color(col[1]) translate([ -$explode, 0, 0 ])
			rotate([ 90, 0, 90 ]) ext() side();
		color(col[1]) translate([ wx-t + $explode, 0, 0 ])
			rotate([ 90, 0, 90 ]) ext() side();
		color(col[2]) translate([ 0, dx + $explode, 0 ])
			rotate([ 90, 0, 0 ]) ext() back();
	}
}


// ************************** main **************************

module demo()
{
	rack("assembled", "A");
	translate([ 0, 0, (82+8+32+8)/2 + $explode*2 ]) rack("assembled", "B", h = 32);
	translate([ $explode*4, 0, 0 ]) assign(xp = $explode*2, $explode = 0) {
		for (x = [-2:+2]) {
			translate([ 2*xp, x*102, 0 ]) box("assembled");
		}
		translate([ 0, -256/2, 130/2 + xp ])
			box("assembled", h = 30, w = 250, s = 10,
				pp = [ 33, 66, 99, 132, 165 ]);
		translate([ 0, 127/2, 130/2 + xp ])
			box("assembled", h = 30, w = 125, s = 10, pp = [ 100 ]);
		translate([ 0, 3*127/2, 130/2 + xp ])
			box("assembled", h = 30, w = 125, s = 10, pp = [ 66, 132 ]);
	}
}

// box("assembled");
// box("parts");

// rack("assembled");
// rack("parts");

demo();

