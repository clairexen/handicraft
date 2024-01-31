
coords = [
	0, // 10
	[ -1, -1, -1 ], // 11
	[  0, -1, -1 ], // 12
	[ +1, -1, -1 ], // 13
	[ -1,  0, -1 ], // 14
	[  0,  0, -1 ], // 15
	[ +1,  0, -1 ], // 16
	[ -1, +1, -1 ], // 17
	[  0, +1, -1 ], // 18
	[ +1, +1, -1 ], // 19
	0, // 20
	[ -1, -1,  0 ], // 21
	[  0, -1,  0 ], // 22
	[ +1, -1,  0 ], // 23
	[ -1,  0,  0 ], // 24
	[  0,  0,  0 ], // 25
	[ +1,  0,  0 ], // 26
	[ -1, +1,  0 ], // 27
	[  0, +1,  0 ], // 28
	[ +1, +1,  0 ], // 29
	0, // 30
	[ -1, -1, +1 ], // 31
	[  0, -1, +1 ], // 32
	[ +1, -1, +1 ], // 33
	[ -1,  0, +1 ], // 34
	[  0,  0, +1 ], // 35
	[ +1,  0, +1 ], // 36
	[ -1, +1, +1 ], // 37
	[  0, +1, +1 ], // 38
	[ +1, +1, +1 ], // 39
];

module tet(A, B, C, D)
{
	X = (coords[A - 10] + coords[B - 10] + coords[C - 10] + coords[D - 10]) * 0.25;
	translate(X * $explode) {
		polyhedron(
			points=[
				coords[A - 10] * 100,
				coords[B - 10] * 100,
				coords[C - 10] * 100,
				coords[D - 10] * 100
			], triangles=[
				[ 0, 1, 2 ],
				[ 0, 3, 1 ],
				[ 0, 2, 3 ],
				[ 1, 3, 2 ]
			]);
	}
}

module tetsplit(A, B, C, D, pAB, pAC, pAD, pBC, pBD, pCD)
{
	if (!pAB && !pAC && !pAD && !pBC && !pBD && !pCD)
		tet(A, B, C, D);
}

module cube(
		p12, p16, p18, p14, p15,
		p21, p22, p23, p26, p29, p28, p27, p24,
		p32, p36, p38, p34, p35
	)
{
	$explode = 50 + 50*cos($t*360);

	//        A   B   C   D   AB,  AC,  AD,  BC,  BD,  CD
	tetsplit(11, 13, 17, 31, p12, p14, p21, p15, p22, p24);
	tetsplit(17, 13, 19, 39, p15, p18, p28, p16, p26, p29);
	tetsplit(31, 39, 33, 13, p35, p32, p22, p36, p26, p23);
	tetsplit(39, 31, 37, 17, p35, p38, p28, p34, p24, p27);
	tetsplit(31, 39, 13, 17, p35, p22, p24, p26, p28, p15);
}

cube(
	false, false, false, false, false,
	false, false, false, false, false, false, false, false,
	false, false, false, false, false
);

