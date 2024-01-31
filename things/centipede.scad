
module servo(angle) {
    color([0.3, 0.3, 1])
        translate([10, -15, -5])
        cube([20, 30, 10]);
    
    color([0.8, 0.8, 0.8])
        rotate([angle, 0, 0]) {
            translate([2, 0, 0])
                rotate([0, 90, 0])
                cylinder(10, 3, 3, $fn=36);
            translate([0, -10, -4])
                cube([5, 20, 8]);
        }
}

module servo_joint(angle,
        r=[0, 0, 0], t=[0, 0, 0]) {
    translate(t)
        rotate(r) {
            rotate([angle, 0, 0])
                children();
            rotate([0, 0, 180])
                servo(-angle);
        }
}

module leg(angles) {
    servo_joint(angles[0]-90) {
        translate([5, 0, -5])
            rotate([0, -90, 0])
            translate([25, 0, 0]) {
                servo_joint(angles[1]-180)
                    rotate ([90, 0, 0]) {
                        translate([0, -5, -11])
                            cube([2, 40, 22]);
                        translate([0, 15, -20])
                            cube([2, 20, 40]);
                        servo_joint(angles[2]-170,
                                r=[90, 180, 0],
                                t=[-20, 25, 0])
                            translate([0, -15, -2])
                            cube([4, 100, 4]);
                    }
            }
    }
}

module segment(angles) {
    translate([50, 30, 0])
        leg([ for (i = [0:2]) angles[i] ]);
    
    translate([-50, 30, 0])
        rotate([0, 0, 180])
        leg([ for (i = [3:5]) -angles[i] ]);

    translate([50, 100, 0])
        leg([ for (i = [6:8]) angles[i] ]);
    
    translate([-50, 100, 0])
        rotate([0, 0, 180])
        leg([ for (i = [9:11]) -angles[i] ]);
    
    translate([50, 170, 0])
        leg([ for (i = [12:14]) angles[i] ]);
    
    translate([-50, 170, 0])
        rotate([0, 0, 180])
        leg([ for (i = [15:17]) -angles[i] ]);
    
    difference() {
        translate([0, 100, 0]) intersection() {
            cylinder(r=105, h=4, center=true, $fn=100);
            cube([70, 250, 4], center=true);
        }
        translate([0, 100, 0])
            cube([25, 180, 10], center=true);
        translate([0, 65, 0])
            cube([50, 30, 10], center=true);
        translate([0, 135, 0])
            cube([50, 30, 10], center=true);
    }
}

angles = [ for (i = [0:17]) 15*sin((10+i)*360*$t) ];
    
segment(angles);
translate([0, 220, 0]) rotate([0, 0, 20]) {
    segment(angles);
    translate([0, 220, 0]) rotate([0, 0, -15]) {
        segment(angles);
        translate([0, 220, 0]) rotate([0, 0, -25]) {
            segment(angles);
        }
    }
}
