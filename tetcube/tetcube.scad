
module tet1()
	polyhedron(
		points=[
			[ 0, 0, 0 ],
			[ 1, 0, 0 ],
			[ 0, 1, 0 ],
			[ 0, 0, 1 ]
		], triangles=[
			[ 0, 1, 2 ],
			[ 0, 3, 1 ],
			[ 0, 2, 3 ],
			[ 1, 3, 2 ]
		]);

module tet2()
	polyhedron(
		points=[
			[ 0, 0, 0 ],
			[ 1, 1, 0 ],
			[ 0, 1, 1 ],
			[ 1, 0, 1 ]
		], triangles=[
			[ 0, 1, 2 ],
			[ 0, 3, 1 ],
			[ 0, 2, 3 ],
			[ 1, 3, 2 ]
		]);

module cube1(s)
{
	translate([-s, -s, -s]) rotate([   0, 0,   0 ]) tet1();
	translate([+s, +s, -s]) rotate([   0, 0, 180 ]) tet1();
	translate([-s, +s, +s]) rotate([ 180, 0,   0 ]) tet1();
	translate([+s, -s, +s]) rotate([ 180, 0, 180 ]) tet1();
	rotate([0, 0, 90]) translate([-0.5, -0.5, -0.5]) tet2();
}

module tet3()
	polyhedron(
		points=[
			[ 0, 0, 0 ],
			[ 1, 0, 0 ],
			[ 0, 1, 1 ],
			[ 0, 0, 1 ]
		], triangles=[
			[ 0, 1, 2 ],
			[ 0, 3, 1 ],
			[ 0, 2, 3 ],
			[ 1, 3, 2 ]
		]);

module tet4()
	polyhedron(
		points=[
			[  0,    0,   0.5 ],
			[  0.5,  0.5, 0   ],
			[ -0.5,  0.5, 0   ],
			[ -0.5, -0.5, 0   ]
		], triangles=[
			[ 0, 2, 1 ],
			[ 0, 1, 3 ],
			[ 0, 3, 2 ],
			[ 1, 2, 3 ]
		]);

module cube2(s)
{
	translate([-s, -s, -0.5]) rotate([ 0, 0,   0 ]) tet3();
	translate([+s, -s, -0.5]) rotate([ 0, 0,  90 ]) tet3();
	translate([+s, +s, -0.5]) rotate([ 0, 0, 180 ]) tet3();
	translate([-s, +s, -0.5]) rotate([ 0, 0, 270 ]) tet3();
	translate([-s/2+0.25, +s/2-0.25, -s]) rotate([ 0, 0,   0 ]) tet4();
	translate([+s/2-0.25, -s/2+0.25, -s]) rotate([ 0, 0, 180 ]) tet4();
	translate([-s/2+0.25, -s/2+0.25, +s]) rotate([ 180, 0,   0 ]) tet4();
	translate([+s/2-0.25, +s/2-0.25, +s]) rotate([ 180, 0, 180 ]) tet4();


}

// all set up:
translate([-50, 0, 0]) scale(50) cube1(0.7);
translate([+50, 0, 0]) scale(50) cube2(0.7);

// seperate parts for printing:
// scale(50) tet1();
// scale(50) rotate(55, [1,1,0]) tet2();
// scale(50) rotate([-45,0,0]) tet3();
// scale(50) tet4();

