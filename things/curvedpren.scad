
base_radius = 1150;
base_width = 280;
base_length = 2 * 650;

belt_width = 70;
belt_length = 180;

hole_inset = 60;
hole_radius = 15;
hole_position = 230;

$fn = 100;

module curved_segment(radius, base_radius, length_left, length_right, width)
{
	length_angle_left = 90 * length_left / (PI * base_radius);
	length_angle_right = 90 * length_right / (PI * base_radius);

	difference() {
		circle(radius + width);
		circle(radius);
		rotate(-length_angle_left) square(radius + 2*width);
		rotate(90+length_angle_right) square(radius + 2*width);
		translate([ -(radius + 2*width), -(radius + 2*width) ]) square([ 2*radius + 4*width, radius + 2*width ]);
	}

}

module curved_band(radius, length_left, length_right, width, curved_left)
{
	length_angle_left = 90 * (length_left - width) / (PI * (radius + width/2));
	length_angle_right = 90 * (length_right - width) / (PI * (radius + width/2));

	curved_segment(radius, radius + width/2, length_left - width, length_right - width, width);

	for (a = [-length_angle_left, +length_angle_right])
		if (a > 0 || curved_left)
			rotate(a) translate([0, radius + width/2]) circle(width / 2);
}

module curvedpren()
{
	cutout_radius = base_width / 2 - belt_width;
	cutout_angle = 90 * (base_length + 2*cutout_radius) / (PI * (base_radius + base_width/2));
	hole_angle = 180 * hole_position / (PI * (base_radius + hole_inset));

	top_length = base_length * base_radius / (base_radius + base_width/2);

	tip_length = top_length;
	tip_angle = 90 * tip_length / (PI * base_radius);

	corner_length = top_length - 2*belt_width;
	corner_angle = 90 * corner_length / (PI * base_radius);

	union()
	{
		rotate(-tip_angle) translate([0, base_radius]) scale([-1, 1]) translate([belt_width, 0]) rotate(90) translate([0, -base_radius])
				curved_band(base_radius, 2*belt_length, 2*belt_length, belt_width, false);

		difference() {
			union() {
				curved_band(base_radius, base_length, base_length, base_width, true);
				curved_segment(base_radius, base_radius + base_width/2, base_length, base_length + 2*cutout_radius, base_width/2);
				curved_band(base_radius, base_length * base_radius / (base_radius + base_width/2), base_length + 2*belt_length, belt_width, false);
			}

			for (a = [-cutout_angle, +cutout_angle])
				rotate(a) translate([0, base_radius + base_width/2]) circle(base_width / 2 - belt_width);
			rotate(-hole_angle) translate([0, base_radius + hole_inset]) circle(hole_radius);
		}

		difference() {
			rotate(-corner_angle) translate([0, base_radius]) circle(belt_width/8);
			rotate(-corner_angle) translate([-belt_width/8, base_radius-belt_width/8]) circle(belt_width/8);
		}
	}
}

render() translate([0, -base_radius]) curvedpren();

