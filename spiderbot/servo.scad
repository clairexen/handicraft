
module servo()
{
	h2 = dxf_dim(file = "servo.dxf", layer = "0", name = "h2");
	h1 = dxf_dim(file = "servo.dxf", layer = "0", name = "h1");

	color([ 0.5, 0.5, 0.5 ])
	linear_extrude(file = "servo.dxf", layer = "axis", origin = [0, 0], height=10);

	color([ 0.5, 0.5, 0.8 ]) translate([ 0, 0, -h1-h2 ])
	linear_extrude(file = "servo.dxf", layer = "back2", origin = [40, 0], height=h2, convexity = 2);

	color([ 0.5, 0.5, 0.8 ]) translate([ 0, 0, -h1 ])
	linear_extrude(file = "servo.dxf", layer = "back1", origin = [20, 0], height=h1, convexity = 2);

	color([ 0.5, 0.5, 0.8 ]) translate([ 0, 0, -24.5 ])
	linear_extrude(file = "servo.dxf", layer = "body", origin = [0, 0], height=27);	
}

