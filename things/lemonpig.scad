$fn = 200;

module pear() {
    rotate([0, 90, 0])
    rotate_extrude() rotate(90) intersection() {
        translate([-70, 0]) square(200);
        difference() {
            union() {
                circle(50);
                intersection() {
                    rotate(20) translate([50, 0]) square(100, center=true);
                    rotate(-20) translate([50, 0]) square(100, center=true);
                }
                translate([93.5, 0]) circle(18);
            }
            translate([-50, 0]) rotate(45) square(15, center=true);
        }
    }
}

module stick() {
    cylinder(h = 100, r = 3);
    sphere(6);
}

module coin() {
    cylinder(h = 5, r = 20);
}

module eye() {
    scale([1, 1, 2]) sphere(6);
}

translate([0, 0, 80]) pear();

translate([0, -50, 0]) rotate([-20, 15, 0]) stick();
translate([0, +50, 0]) rotate([+20, 15, 0]) stick();

translate([90, -50, 0]) rotate([-20, -15, 0]) stick();
translate([90, +50, 0]) rotate([+20, -15, 0]) stick();

translate([-45, 0, 60]) rotate([10, 0, 0]) coin();

translate([-40, -25, 100]) rotate([+10, -30, +10]) eye();
translate([-40, +25, 100]) rotate([-10, -30, -10]) eye();

translate([100, 0, 80]) rotate([-10, 40, 20]) stick();