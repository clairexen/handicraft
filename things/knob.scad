disc_r = 25;
disc_t = 3;

dial_r = 16;
dial_t = 5;
dial_h = 14;

post_h = 17;
post_r = 8;

ring_r = 12;
ring_h = 2;

hole_r = 3;
hole_f = 2;

cutout_offset = 10;
cutout_angle = 29;

square_w = 25;

$fn = 100;
eps = 0.2;

module top() {
    difference() {
        cylinder(r = 25, h = 3);
        translate([-(square_w+eps)/2, -(square_w+eps)/2, -eps])
            cube([square_w+eps, square_w+eps, disc_t+2*eps]);
    }

    intersection() {
        translate([-dial_r, -dial_t/2, disc_t-eps])
            cube([2*dial_r, dial_t, dial_h]);
        translate([0, 0, cutout_offset]) union() {
            rotate([0, -cutout_angle, 0]) translate([-2*dial_r, -dial_t/2, 0])
                cube([4*dial_r, dial_t, dial_h]);
            rotate([0, +cutout_angle, 0]) translate([-2*dial_r, -dial_t/2, 0])
                cube([4*dial_r, dial_t, dial_h]);
            
        }
    }
}

module bottom() {
    difference() {
        cylinder(r = post_r, h = post_h+eps);
        translate([0, 0, -eps]) {
            difference() {
                cylinder(r = hole_r, h = post_h+eps);
                translate([hole_f, -hole_r, -eps])
                    cube([hole_r, 2*hole_r, post_h+3*eps]);
            }
        }
    }

    translate([0, 0, post_h-ring_h])
        cylinder(r1 = post_r, r2 = ring_r, h = ring_h+eps);
    
    translate([-square_w/2, -square_w/2, post_h])
        cube([square_w, square_w, disc_t]);
}

module knob() {
    top();
    translate([0, 0, -post_h])
        bottom();
}

module parts() {
    translate([disc_r+2, 0, 0]) top();
    translate([-square_w/2-2, 0, post_h+disc_t]) rotate([180, 0, 0]) bottom();
}

// knob();
parts();
