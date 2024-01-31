
dim_width = 200;
dim_height = 450;
dim_window = 130;
dim_winpos = [ 140, 330 ];
dim_teeth = 20;
dim_thick = 6;
dim_foots = 40;

colors = [ "orange", "green", "orange", "green", "blue" ];
explode = 0;

module top_stage0()
{
	translate([-dim_width/2, -dim_width/2])
	difference() {
		square(dim_width);
		for (i = [0:dim_width/(2*dim_teeth)-1])
			translate([0, i*2*dim_teeth]) square([dim_thick, dim_teeth]);
	}
}

module top_stage1()
{
	intersection_for(angle = [0:90:270])
		rotate(angle) top_stage0();
}

module side_stage0()
{
	difference() {
		translate([-dim_width/2, 0]) square([dim_width, dim_height]);
		for (p = dim_winpos)
			translate([0, p]) square(dim_window, center = true);
		for (i = [1:dim_height/(2*dim_teeth)-1])
			translate([-dim_width/2, i*2*dim_teeth]) square([dim_thick, dim_teeth]);
		square([dim_width - dim_foots, 2*dim_foots], center = true);
	}
}

module side_stage1()
{
	difference() {
		side_stage0();
		translate([dim_width - dim_thick, 0]) side_stage0();
		translate([0, dim_height + dim_width/2 - dim_thick]) top_stage1();
	}
}

module cuttr()
{
	for (i = [0:3])
		translate([(dim_width+dim_thick)*i, 0]) side_stage1();
	translate([(dim_width+dim_thick)*4, dim_height/2]) top_stage1();
}

module model()
{
	for (i = [0:3])
		rotate([0, 0, 90*i]) translate([0, -dim_width/2 + dim_thick - explode, 0])
		rotate([90, 0, 0]) color(colors[i]) linear_extrude(dim_thick) side_stage1();
	translate([0, 0, dim_height - dim_thick + explode])
		color(colors[4]) linear_extrude(dim_thick) top_stage1();
}

// cuttr();
model();

