R =  [1, 0, 0];
L = -[1, 0, 0];
B =  [0, 1, 0];
F = -[0, 1, 0];
U =  [0, 0, 1];
D = -[0, 0, 1];

module showPoint(p) {
	# translate(p) sphere(30);
}

module simpleBar(w, d, xA=0, xB=0) {
	l = sqrt(d[0]^2 + d[1]^2 + d[2]^2);

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
}

module bar(w, pA, pB, eA=[], eB=[], name = "UNNAMED") {
	d = pB - pA;
	l = sqrt(d[0]^2 + d[1]^2 + d[2]^2);

	x = 1000; // Large enough to effectively be a half-space
	xA = len(eA) ? x : 0;
	xB = len(eB) ? x : 0;

	translate(pA) render(convexity=1) difference() {
		simpleBar(w, d, xA, xB);

		for (i = [0:1:len(eA)-1])
			simpleBar(2*(x+w), (x+w)*eA[i]);

		for (i = [0:1:len(eB)-1])
			translate(d) simpleBar(2*(x+w), (x+w)*eB[i]);
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
W = 44;

c1 = [1.0, 0.8, 0.8];
c2 = [1.0, 1.0, 0.5];
c3 = [0.8, 0.8, 0.8];
c4 = [0.5, 0.8, 0.5];
c5 = [0.8, 0.5, 0.5, 0.2];
c6 = [0.5, 0.5, 0.8, 0.2];

height = 2000;
width = 2400;
depth = 2100;
indent = 600;
ladderWidth = 500;
strutSpan = 400;

origin = [0, 0, 0];
// showPoint(origin);

module foundation(O) {

	module ladder() {
		p1 = O + W/2*(R+B) + W*R;
		p2 = p1 + height*U;

		for (q = [0, ladderWidth-W])
			color(c1) translate(q*R) bar(W, p1, p2, [], [], "LADDER");

		N = 10;
		for (i = [1:9])
			color(c4) translate(W*R + W/2*(R+B) + i*height/N*U)
			rotate([0, 90, 0]) cylinder(ladderWidth - W, d=0.75*W);
	}

	module legs() {
		h = height + 2*W;
		p1 =  O + 1.5*W*(R+B);
		p2 = p1 + (depth-3*W)*B;
		p3 = p2 + (width-3*W)*R;
		p4 = p3 + (depth-3*W-indent)*F;

		for (p = [p2, p3, p4])
			color(c3) bar(W, p, p + h*U, [], [], "LEG");

		// bottom struts
		p5 = p3 + W/2*(B+L+U) + W*U;
		color(c1) bar(W, p4 + W/2*(F+U) + W*L, p3 + W/2*(B+U) + W*L, [], [], "BOTTOM");
		color(c2) bar(W, p3 + W/2*(R+U) + W*(B+U), p2 + W/2*(L+U) + W*(B+U), [], [], "BOTTOM");
		color(c4) bar(W, p5 + strutSpan*F, p5 + strutSpan*L, [R], [B], "BOT_STRUT");

	}

	module frame() {
		p1 = O + W/2*(U+B) + height*U;
		p2 = p1 + width*R;

		p1t = p1 + (depth-W)*B;
		p2t = p2 + (depth-W)*B;

		color(c2) bar(W, p1, p2, [], [], "FRAME");
		color(c2) bar(W, p1t, p2t, [], [], "FRAME");

		p3 = p1 + W*U + W/2*(F+R);
		p4 = p3 + depth*B;

		p3t = p3 + (width-W)*R;
		p4t = p4 + (width-W)*R;

		color(c1) bar(W, p3, p4, [], [], "FRAME");
		color(c1) bar(W, p3t, p4t, [], [], "FRAME");

		// horizontal struts
		color(c4) bar(W, p1 + strutSpan*B, p3 + strutSpan*R + W*(B+D), [L], [F], "HORIZ_STRUT");
		color(c4) bar(W, p1t + strutSpan*F, p4 + strutSpan*R + W*(F+D), [L], [B], "HORIZ_STRUT");
		color(c4) bar(W, p2 + strutSpan*B, p3t + strutSpan*L + W*(B+D), [R], [F], "HORIZ_STRUT");
		color(c4) bar(W, p2t + strutSpan*F, p4t + strutSpan*L + W*(F+D), [R], [B], "HORIZ_STRUT");

		// vertical struts left and right
		p5 = p3 + W/2*D;
		p6 = p4 + W/2*D + W*F;
		p7 = p5 + (width-W)*R + (indent+W)*B;
		p8 = p6 + (width-W)*R;

		color(c4) bar(W, p5 + 1.5*strutSpan*D, p5 + 1.5*strutSpan*B, [F], [U], "VERT_STRUT");
		color(c4) bar(W, p6 + 1.5*strutSpan*D, p6 + 1.5*strutSpan*F, [B], [U], "VERT_STRUT");
		color(c4) bar(W, p7 + 1.5*strutSpan*D, p7 + 1.5*strutSpan*B, [F], [U], "VERT_STRUT");
		color(c4) bar(W, p8 + 1.5*strutSpan*D, p8 + 1.5*strutSpan*F, [B], [U], "VERT_STRUT");

		// vertical struts front and back
		p9 = p1 + W/2*D + (W+ladderWidth)*R;
		p10 = p1t + W/2*D + W*R;
		p11 = p2t + W/2*D + W*L;

		color(c4) bar(W, p9 + strutSpan*D, p9 + strutSpan*R, [L], [U], "VERT_STRUT");
		color(c4) bar(W, p10 + 2*strutSpan*D, p10 + 2*strutSpan*R, [L], [U], "VERT_STRUT");
		color(c4) bar(W, p11 + 2*strutSpan*D, p11 + 2*strutSpan*L, [R], [U], "VERT_STRUT");
	}

	legs();
	ladder();
	frame();
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

foundation(origin);
// horzgrid();
