
include <servo.scad>;
include <leg.scad>;
include <head.scad>;

module spider(pos)
{
	axis0 = dxf_cross(file = "body.dxf", layer = "axis0");
	axis1 = dxf_cross(file = "body.dxf", layer = "axis1");
	axis2 = dxf_cross(file = "body.dxf", layer = "axis2");
	axis3 = dxf_cross(file = "body.dxf", layer = "axis3");
	axis4 = dxf_cross(file = "body.dxf", layer = "axis4");
	axis5 = dxf_cross(file = "body.dxf", layer = "axis5");
	axis6 = dxf_cross(file = "body.dxf", layer = "axis6");

	linear_extrude(file = "body.dxf", layer = "layer1", height = 5, convexity = 2);

	translate([ 0, 0, -45 ])
	linear_extrude(file = "body.dxf", layer = "layer2", origin = [ 200, 0 ], height = 5, convexity = 2);

	translate([ axis0[0], axis0[1], -5 ]) rotate(90)
		head(pos[0][0], pos[0][1]);

	translate([ axis1[0], axis1[1], -5 ]) rotate(45)
		mleg(pos[1][0], pos[1][1], pos[1][2]);

	translate([ axis2[0], axis2[1], -5 ]) rotate(0)
		mleg(pos[2][0], pos[2][1], pos[2][2]);

	translate([ axis3[0], axis3[1], -5 ]) rotate(-45)
		mleg(pos[3][0], pos[3][1], pos[3][2]);

	translate([ axis4[0], axis4[1], -5 ]) rotate(135)
		leg(pos[4][0], pos[4][1], pos[4][2]);

	translate([ axis5[0], axis5[1], -5 ]) rotate(180)
		leg(pos[5][0], pos[5][1], pos[5][2]);

	translate([ axis6[0], axis6[1], -5 ]) rotate(-135)
		leg(pos[6][0], pos[6][1], pos[6][2]);	
}

if ($t == 0)
	spider([
		[ 0, 0 ],
		[ 0, 30, 40 ],
		[ 0, 30, 40 ],
		[ 0, 30, 40 ],
		[ 0, 30, 40 ],
		[ 0, 30, 40 ],
		[ 0, 30, 40 ],
	]);

include <testdata.scad>;
if ($t > 0)
	spider(spider_config($t));
