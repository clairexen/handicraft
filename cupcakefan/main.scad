
$debug = false;

module invader()
{
	module outer()
	render(convexity = 10)
	intersection()
	{
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "front",
			origin = [0, 0], height = 100, convexity = 2, center = true);
	
		translate([0, 0, -10])
		dxf_linear_extrude(file = "invader.dxf", layer = "top",
			origin = [0, -100], height = 100, convexity = 2, center = false);
	
		rotate(90, [0, 1, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "side",
			origin = [-100, -100], height = 150, convexity = 2, center = false);
	}

	module inner()
	render(convexity = 10)
	intersection()
	{
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "front_cutout",
			origin = [0, 0], height = 100, convexity = 2, center = true);
	
		dxf_linear_extrude(file = "invader.dxf", layer = "top_cutout",
			origin = [0, -100], height = 100, convexity = 2, center = false);
	
		rotate(90, [0, 1, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "side_cutout",
			origin = [-100, -100], height = 150, convexity = 2, center = false);
	}

	module exit()
	render(convexity = 10)
	intersection()
	{
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "front_exit",
			origin = [0, 0], height = 100, convexity = 2, center = true);
	
		rotate(90, [0, 1, 0])
		dxf_linear_extrude(file = "invader.dxf", layer = "side_exit",
			origin = [-100, -100], height = 150, convexity = 2, center = false);
	}

	difference()
	{
		outer();
		inner();
		exit();

		if ($debug) {
			translate([15, 32, 8]) cube(100);
			translate([93, -35, -5]) cube(30);
		}
	}
}

module adapter()
{
	module outer()
	render(convexity = 10)
	intersection()
	{
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "adapter.dxf", layer = "front",
			origin = [0, 0], height = 100, convexity = 2, center = true);

		translate([0, 0, -30])
		dxf_linear_extrude(file = "adapter.dxf", layer = "top",
			origin = [0, -100], height = 150, convexity = 2, center = false);
	
		rotate(90, [0, 1, 0])
		dxf_linear_extrude(file = "adapter.dxf", layer = "side",
			origin = [-100, -100], height = 150, convexity = 2, center = true);
	}

	module inner()
	render(convexity = 10)
	intersection()
	{
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "adapter.dxf", layer = "front_cutout",
			origin = [0, 0], height = 100, convexity = 2, center = true);

		translate([0, 0, -30])
		dxf_linear_extrude(file = "adapter.dxf", layer = "top_cutout",
			origin = [0, -100], height = 150, convexity = 2, center = false);
	
		rotate(90, [0, 1, 0])
		dxf_linear_extrude(file = "adapter.dxf", layer = "side_cutout",
			origin = [-100, -100], height = 150, convexity = 2, center = true);
	}

	difference()
	{
		outer();
		inner();
	}
}

module dino_left()
{
	translate([0, +14.4999, +5])
	{
		translate([0, -5, 0])
		rotate(180, [0, 0, 1])
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "dinos.dxf", layer = "wdb",
			origin = [+100, 0], height = 5, convexity = 10);
	
		translate([0, -24.5002, 0])
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "dinos.dxf", layer = "wdf",
			origin = [0, 0], height = 5, convexity = 10);
	
		translate([0, 0, -10])
		dxf_linear_extrude(file = "dinos.dxf", layer = "pl",
			origin = [0, -100], height = 10, convexity = 10);
	}
}

module dino_right()
{
	translate([0, +14.4999, +5])
	{
		translate([0, -5, 0])
		rotate(180, [0, 0, 1])
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "dinos.dxf", layer = "bdb",
			origin = [-100, 0], height = 5, convexity = 10);
	
		translate([0, -24.5002, 0])
		rotate(90, [1, 0, 0])
		dxf_linear_extrude(file = "dinos.dxf", layer = "bdf",
			origin = [+200, 0], height = 5, convexity = 10);
	
		translate([0, 0, -10])
		dxf_linear_extrude(file = "dinos.dxf", layer = "pr",
			origin = [100, -100], height = 10, convexity = 10);
	}
}

module zstage()
{
	translate([-50, 0, 0])
	dino_left();

	translate([+50, 0, 0])
	dino_right();

	translate([0, 0, -10])
	dxf_linear_extrude(file = "zstage.dxf",
		height = 5, convexity = 10);
}

module fan() { <fan.scad> }

module full_setup()
{
	translate([-95, 0, -55])
	invader();

	translate([-61, +23, +21])
	rotate(180, [0, 0, 1])
	adapter();
	
	% translate([0, 0, -50])
	zstage();
	
	% translate([0, 0, -10])
	dxf_linear_extrude(file = "top.dxf", height = 5,
		origin = [+(10.0002 + 130), +(84.9968 + 35.0585)],
		convexity = 10);

	% translate([-75, 42, 51])
	rotate(90, [1, 0, 0])
	fan();
}

module solid_part()
{
	$debug = true;

	difference()
	{
		union()
		{
			translate([-60, 0, -20])
			invader();
		
			translate([-26, +23, +56])
			rotate(180, [0, 0, 1])
			adapter();
		}

		translate([-80, -45, 30]) rotate(45, [0, 0, 1]) cube(60);
	}
}

module part1()
{
	translate([0, +45, 0])
	rotate(90, [0, -1, 0])
	intersection()
	{
		invader();
		translate([-10, -50, -10]) cube([150, 50, 49.9]);
	}	
}

module part2()
{
	rotate(90, [0, -1, 0])
	intersection()
	{
		invader();
		translate([-10, 0, -10]) cube([150, 50, 49.9]);
	}	
}

module part3()
{
	// rotate(45, [0, 0, 1])
	translate([10, 0, -40.1])
	intersection()
	{
		invader();
		translate([-10, -50, 40.1]) cube([150, 100, 50]);
	}	
}

module part4()
{
	translate([-40, -10, +6])
	rotate(-90, [0, 0, 1])
	adapter();
}

full_setup();
//solid_part();
//part1();
//part2();
//part3();
//part4();
