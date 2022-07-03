R =  [1, 0, 0];
L = -[1, 0, 0];
B =  [0, 1, 0];
F = -[0, 1, 0];
U =  [0, 0, 1];
D = -[0, 0, 1];

module showPoint(p) {
	# translate(p) sphere(30);
}

module bar(w, pA, pB, eA=[], eB=[], name = "UNNAMED") {
	d = pB - pA;
	l = sqrt(d[0]^2 + d[1]^2 + d[2]^2);

	x = 1000; // Large enough to effectively be a half-space
	xA = len(eA) ? x : 0;
	xB = len(eB) ? x : 0;

	translate(pA) render(convexity=1) difference() {
		if (abs(d[0]) >= abs(d[1]) && abs(d[0]) >= abs(d[2])) {
			rotate([0, -atan2(d[2], sqrt(d[1]^2 + d[0]^2)), atan2(d[1], d[0])])
					translate([-xA, -w/2, -w/2]) cube([l+xA+xB, w, w]);
		} else
		if (abs(d[1]) >= abs(d[0]) && abs(d[1]) >= abs(d[2])) {
			rotate([atan2(d[2], sqrt(d[0]^2 + d[1]^2)), 0, -atan2(d[0], d[1])])
					translate([-w/2, -xA, -w/2]) cube([w, l+xA+xB, w]);
		} else {
			rotate([-atan2(d[1], sqrt(d[0]^2 + d[2]^2)), atan2(d[0], d[2]), 0])
					translate([-w/2, -w/2, -xA]) cube([w, w, l+xA+xB]);
		}

		/*
		translate([-x-w, -y, -y]) cube([x+w, 2*y, 2*y]);
		translate(d-[0,y,y]) cube([x+w, 2*y, 2*y]);
		*/
	}

	echo(str("Bar ", name, ": length=", l));
}

module barTest() {
	for (p = [L, R, D, U, F, B]) {
		pA = 300 * p;  // showPoint(pA);
		pB = 600 * p;  // showPoint(pB);
		color([1.0, 0.5, 0.5]) bar(30, pA, pB, [], []);
	}

	for (alpha = [30, 60, 120, 150, 210, 240, 300, 330]) {
		p = [cos(alpha), sin(alpha), 0];
		pA = 300 * p;  // showPoint(pA);
		pB = 500 * p;  // showPoint(pB);
		color([1.0, 1.0, 0.3]) bar(20, pA, pB, [], []);
	}

	for (alpha = [30, 60, 120, 150, 210, 240, 300, 330]) {
		p = [cos(alpha), 0, sin(alpha)];
		pA = 300 * p;  // showPoint(pA);
		pB = 500 * p;  // showPoint(pB);
		color([1.0, 0.3, 1.0]) bar(20, pA, pB, [], []);
	}

	for (alpha = [30, 60, 120, 150, 210, 240, 300, 330]) {
		p = [0, cos(alpha), sin(alpha)];
		pA = 300 * p;  // showPoint(pA);
		pB = 500 * p;  // showPoint(pB);
		color([0.3, 1.0, 1.0]) bar(20, pA, pB, [], []);
	}

	for (pX = [-1, 1]/sqrt(3), pY = [-1, 1]/sqrt(3), pZ = [-1, 1]/sqrt(3)) {
		p = [pX, pY, pZ];
		pA = 300 * p;  // showPoint(pA);
		pB = 400 * p;  // showPoint(pB);
		color([0.5, 1.0, 0.5]) bar(40, pA, pB, [], []);
	}
}

// barTest();

w0 = 20;
w1 = 44;
w2 = 57;

c1 = [1.0, 0.8, 0.8];
c2 = [1.0, 1.0, 0.5];
c3 = [0.8, 0.8, 0.8];
c4 = [0.5, 0.8, 0.5];
c5 = [0.8, 0.5, 0.5, 0.2];
c6 = [0.5, 0.5, 0.8, 0.2];

height = 2000;
width = 2400;
depth = 2150;
ladderWidth = 400;

origin = [0, 0, 0];
// showPoint(origin);

module foundation() {
	pDL = origin + w2/2*(R+B);
	pUL = pDL + height*U;
	// showPoint(pDL);
	// showPoint(pUL);

	pDLR = pDL + ladderWidth*R;
	pULR = pUL + ladderWidth*R;
	// showPoint(pDLR);
	// showPoint(pULR);

	color(c1) bar(w2, pDL, pUL, [], []);
	color(c1) bar(w2, pDLR, pULR, [], []);

	pDR = pDL + (width - w2)*R;
	pUR = pUL + (width - w2)*R;

	color(c1) bar(w2, pDR, pUR, [], []);
	color(c2) bar(w2, pUL + [-w2/2, 0, w2/2], pUR + [w2/2, 0, w2/2], [], []);

	pULB = pUL + (depth - w2)*B;
	pURB = pUR + (depth - w2)*B;

	color(c2) bar(w2, pULB + [-w2/2, 0, w2/2], pURB + [w2/2, 0, w2/2], [], []);
}

module horzgrid() {
	pLF = origin + w1/2*(R+U) + (height+w2)*U;
	pRF = pLF + (width-w1)*R;
	pLB = pLF + depth*B;
	pRB = pRF + depth*B;

	color(c1) bar(w1, pLF, pLB, [], []);
	color(c1) bar(w1, pRF, pRB, [], []);

	N = 6;
	d = (width-w1)/(N+1);

	for (i = [1:N]) {
		color(c3) bar(w1, pLF + (i*d)*R, pLB + (i*d)*R, [], []);
	}

	for (i = [0:N]) {
		color(c4) bar(w1, pLF + (i*d)*R + [w1/2, w1/2, 0], pLF + (i*d+d)*R + [-w1/2, w1/2, 0], [], []);
		color(c4) bar(w1, pLB + (i*d)*R + [w1/2, -w1/2, 0], pLB + (i*d+d)*R + [-w1/2, -w1/2, 0], [], []);
	}

	color(c2) bar(w1, pLF + [-w1/2, w1/2, w1], pRF + [w1/2, w1/2, w1], [], []);
	color(c2) bar(w1, pLB + [-w1/2, -w1/2, w1], pRB + [w1/2, -w1/2, w1], [], []);

	M = 8;
	gap = 50;

	for (i = [1:M]) {
		u = (i-1)*(depth-2*w1+gap)/M;
		v = i*(depth-2*w1+gap)/M - gap;
		p1 = pLF + u*B + [-w1/2, w1, w1/2];
		p2 = pLF + u*B + [-w1/2, w1, w1/2] + (2*d+w1/2)*R;
		p3 = pRF + u*B + [+w1/2, w1, w1/2] - (2*d+w1/2)*R;
		p4 = pRF + u*B + [+w1/2, w1, w1/2];

		color(i%2 ? c6 : c5) translate(p1) cube((p2-p1) + (v-u)*B + 20*U);
		color(i%2 ? c5 : c6) translate(p2) cube((p3-p2) + (v-u)*B + 20*U);
		color(i%2 ? c6 : c5) translate(p3) cube((p4-p3) + (v-u)*B + 20*U);

		if (i < M) {
			color(c4) translate(p1 + (v-u)*B) cube([w1, gap, 20]);
			color(c4) translate(p2 + (v-u)*B + w1/2*L) cube([w1, gap, 20]);
			color(c4) translate(p3 + (v-u)*B + w1/2*L) cube([w1, gap, 20]);
			color(c4) translate(p4 + (v-u)*B + w1*L) cube([w1, gap, 20]);
		}
	}
}

foundation();
horzgrid();
