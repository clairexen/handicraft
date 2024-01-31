
module a()
dxf_linear_extrude(file = "cabinet.dxf", layer = "arduino",
height = 25, convexity = 5);

module p1()
dxf_linear_extrude(file = "cabinet.dxf", layer = "p1",
height = 5, convexity = 5, origin = [ 0, 25 ]);

module p2()
dxf_linear_extrude(file = "cabinet.dxf", layer = "p2",
height = 5, convexity = 5);

module p3()
dxf_linear_extrude(file = "cabinet.dxf", layer = "p3",
height = 5, convexity = 5, origin = [ 0, -85 ]);

module p4()
dxf_linear_extrude(file = "cabinet.dxf", layer = "p4",
height = 5, convexity = 5, origin = [ 105, 0 ]);

translate([-50, 30, -20])
{
	%translate([0, 0, 5])
	a();

	translate([0, 15, 0])
	rotate(90, [1, 0, 0])
	p1();
	
	translate([0, 0, -5])
	p2();
	
	translate([0, -75, 0])
	rotate(-90, [1, 0, 0])
	p3();
	
	translate([95, 0, 0])
	rotate(-90, [0, 1, 0])
	p4();
}
