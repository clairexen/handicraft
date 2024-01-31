
$fa = 10;
$fs = 0.5;
$fn = 0;

module dryerpart(l1 = 5, b1 = 10, l2 = 5, l3 = 35, b3 = 13, b4 = 3,
		h = 5, x = 3, y = 6)
{
	intersection()
	{
		union()
		{
			translate([ 0, -b1/2, 0 ])
			cube([ l1, b1, h ]);
		
			translate([ l1-1, 0, h/2 ])
			rotate([ 0, 90, 0 ])
			cylinder(h = l2+2, r = h/2);
		
			translate([ l1+l2, -b3/2, 0 ])
			polyhedron(
				points = [
					[ 0, b3, h ],
					[ 0, b3, 0 ],
					[ 0, 0, 0 ],
					[ 0, 0, h ],
					[ l3, b3/2+b4/2, h ],
					[ l3, b3/2+b4/2, 0 ],
					[ l3, b3/2-b4/2, h ],
					[ l3, b3/2-b4/2, 0 ],
				],
				triangles = [
					[ 3, 2, 1, 0 ],
					[ 0, 1, 5, 4 ],
					[ 2, 3, 6, 7 ],
					[ 1, 2, 7, 5 ],
					[ 0, 4, 6, 3 ],
					[ 4, 5, 7, 6 ]
				]
			);
		}
		union()
		{
			cube([ (l1+l2+x)*2, (b1+b3)*2, h*3 ], center = true);
			cube([ (l1+l2+l3+x)*2, y, h*3 ], center = true);

		}
	}
}

translate([ -23, -7, 0])
dryerpart(l2 = 4.5);

translate([ +22.25, +7, 0])
rotate([0, 0, 180])
dryerpart(l2 = 4);

