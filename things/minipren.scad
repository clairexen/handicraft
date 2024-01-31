
base_radius = 1150;
base_width = 280;
base_length = 2 * 650;

belt_width = 70;
belt_length = 180;

hole_inset = 60;
hole_radius = 15;
hole_position = 230;

$fn = 100;

module curved_segment(radius, base_radius, length, width)
{
	length_angle = 90 * length / (PI * base_radius);

	difference() {
		circle(radius + width);
		circle(radius);
		rotate(-length_angle) square(radius + 2*width);
		rotate(90+length_angle) square(radius + 2*width);
		translate([ -(radius + 2*width), -(radius + 2*width) ]) square([ 2*radius + 4*width, radius + 2*width ]);
	}

}

module curved_band(radius, length, width)
{
	length_angle = 90 * (length - width) / (PI * (radius + width/2));

	curved_segment(radius, radius + width/2, length - width, width);

	for (a = [-length_angle, +length_angle])
		rotate(a) translate([0, radius + width/2]) circle(width / 2);
}

module minipren()
{
	cutout_radius = base_width / 2 - belt_width;
	cutout_angle = 90 * (base_length + 2*cutout_radius) / (PI * (base_radius + base_width/2));
	hole_angle = 180 * hole_position / (PI * (base_radius + hole_inset));

	difference() {
		union() {
			curved_band(base_radius, base_length, base_width);
			curved_segment(base_radius, base_radius + base_width/2, base_length + 2*cutout_radius, base_width/2);
			curved_band(base_radius, base_length + 2*belt_length, belt_width);
		}

		for (a = [-cutout_angle, +cutout_angle])
			rotate(a) translate([0, base_radius + base_width/2]) circle(base_width / 2 - belt_width);
		rotate(-hole_angle) translate([0, base_radius + hole_inset]) circle(hole_radius);
	}
}

render() translate([0, -base_radius]) minipren();

