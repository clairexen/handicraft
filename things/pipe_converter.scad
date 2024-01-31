
eps = 0.001;
diameter_bottom = 8.3; //8.6
diameter_top = 5.2; // 5.4
thickness = 1.5;

height_bottom = 3.0;
height_mid = 2.0;
height_top = 2.0;

rills_width = 0.5;
rills_depth_bottom = 0.4;
rills_depth_top = 0.2;
rills_twist = 150;
rills_num = 5;
rills_enable = false;

inset_top = 0.5;
inset_bottom = 0.5;

module rills()
{
	difference() {
		translate([0, 0, -eps]) linear_extrude(height = height_bottom + height_mid + height_top + 2*eps, twist = -rills_twist, convexity = 10, slices = 100)
			difference() {
				for (i = [1:rills_num]) rotate(i * 360 / rills_num) translate([0, -rills_width/2]) square([diameter_bottom/2 + eps, rills_width]);
				circle(r = diameter_top/2 - 2*rills_depth_top);
			}
		translate([0, 0, -2*eps]) cylinder(r1 = diameter_bottom/2, r2 = diameter_bottom/2 - thickness, h = 2*thickness);
		translate([0, 0, height_bottom + height_mid + height_top + 2*eps - 2*thickness]) cylinder(r1 = diameter_top/2 - thickness, r2 = diameter_top/2, h = 2*thickness);
	}
}

module outer()
{
	cylinder(r = diameter_bottom/2 + thickness, h = height_bottom);
	translate([0, 0, height_bottom - eps]) cylinder(r1 = diameter_bottom/2 + thickness, r2 = diameter_top/2 + thickness, h = height_mid);
	cylinder(r = diameter_top/2 + thickness, h = height_bottom + height_mid + height_top);
}

module inner()
{
	difference() {
		union() {
			translate([0, 0, -eps]) cylinder(r = diameter_bottom/2, h = height_bottom);
			translate([0, 0, height_bottom - 2*eps]) cylinder(r1 = diameter_bottom/2, r2 = diameter_top/2, h = height_mid);
			translate([0, 0, -eps]) cylinder(r = diameter_top/2, h = height_bottom + height_mid + height_top + 2*eps);
		}
		if (rills_enable)
			rills();
	}

	if (rills_enable) {
		translate([0, 0, -eps]) cylinder(r = diameter_bottom/2 - rills_depth_bottom, h = height_bottom);
		translate([0, 0, height_bottom - 2*eps]) cylinder(r1 = diameter_bottom/2 - rills_depth_bottom, r2 = diameter_top/2 - rills_depth_top, h = height_mid);
		translate([0, 0, -eps]) cylinder(r = diameter_top/2 - rills_depth_top, h = height_bottom + height_mid + height_top + 2*eps);
	}
}

module thing()
{
	$fn = 72;
	difference() {
		outer();
		inner();
		translate([0, 0, -eps]) cylinder(r1 = diameter_bottom/2 + inset_bottom, r2 = diameter_bottom/2 - inset_bottom, h = 2 * inset_bottom);
		translate([0, 0, height_bottom + height_mid + height_top - 2*inset_top + eps]) cylinder(r1 = diameter_top/2 - inset_top, r2 = diameter_top/2 + inset_top, h = 2 * inset_top);
	}
}

difference() {
	!thing();
	rotate([0, 0, 30]) translate([0, 0, -eps]) cube(30);
}

