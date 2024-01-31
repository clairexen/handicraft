
// room size
w = 199;
h = 280;

module room() {
    color([0.8, 0.8, 0.8]) render(convexity=2) difference() {
        translate([180-w-5, -100, -5]) cube([w+10, 210+5+100, h+10]);
        translate([180-w, -100, 0]) cube([w, 210+100, h]);
        
        translate([180-w-5, -100, 30]) cube([w+10, 210+5+100, h-60]);
    }
}

module bed() {
    color([0.8, 0.5, 0.5]) render(convexity=2) difference()
    {
        cube([180, 200, 40]);

        translate([  0, 0, 0]) cube([  1, 200, 20]);
        translate([  4, 0, 0]) cube([ 82, 200, 20]);
        translate([ 89, 0, 0]) cube([  2, 200, 20]);
        translate([ 94, 0, 0]) cube([ 82, 200, 20]);
        translate([179, 0, 0]) cube([  1, 200, 20]);

        translate([0,   0, 0]) cube([180,   1, 20]);
        translate([0,   4, 0]) cube([180, 192, 20]);
        translate([0, 199, 0]) cube([180,   1, 20]);
    }
}

// outside coordinates for frame
x1 = 179;
y1 = -10;
x2 = 181-w;
y2 = 209;

// left indent
li = 100;

// right indent
ri = 100;

// anchor point
px = x2+70;
py = y1+70;
pz = 250;

module framex() {
    color("#aa9988") {
        translate([x1-5, y1,   0]) cube([5, 5, 200]);
        translate([x1-5, y2-5, 0]) cube([5, 5, 200]);
        translate([x2,   y2-5, 0]) cube([5, 5, 200]);
        translate([x2,   y1+li,0]) cube([5, 5, 200]);

        
        translate([x2, y1+li, 120]) cube([5,y2-y1-li,5]);
        translate([x2, y1,    195]) cube([5,y2-y1,5]);

        translate([x1-5, y1, 120]) cube([5,y2-y1,5]);
        translate([x1-5, y1, 195]) cube([5,y2-y1,5]);

        translate([x2, y2-5, 120]) cube([x1-x2,5,5]);
        translate([x2, y2-5, 195]) cube([x1-x2,5,5]);
    }
    
    color("#aaccaa") {
        translate([x2+1, y2-20-1, 125]) cube([x1-x2-2,20,1]);
        translate([x2+1, y2-80-1, 200]) cube([x1-x2-2,80,1]);
    }
}

module frame() {
    module batten58(t, r) {
        translate(t) rotate(r) cube([5.8, 5.8, 250]);
    }

    batten58([x1, y1], 90);
    batten58([x2+ri, y1], 0);
    batten58([x2, y1+li], 0);
    batten58([x2, y2], -90);
    batten58([x1, y2], 180);

    module battenhorz(p1, p2, h) {
        blen = pow(pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2), 0.5);
        brot = 180-atan2(p1[0]-p2[0], p1[1]-p2[1]);
    
        translate([p1[0], p1[1], h]) rotate(brot)
            translate([-5.8/2, -2, 0]) cube([5.8, blen+4, 5.8]);
        echo("batten length", blen);
    }
    
    tp1 = [x2+ri+5.8/2, y1+5.8/2];
    tp2 = [x2+5.8/2, y1+li+5.8/2];

    battenhorz([x1-5.8/2, y1+5.8/2], tp1, 250+5.8);
    battenhorz([x2+5.8/2, y2-5.8/2], tp2, 250+5.8);
    battenhorz([x1-5.8/2, y1+5.8/2], [x1-5.8/2, y2-5.8/2], 250);
    battenhorz([x2+5.8/2, y2-5.8/2], [x1-5.8/2, y2-5.8/2], 250);

    battenhorz(tp1, tp2, 250);
    battenhorz((tp1 + tp2) / 2, [x1-5.8/2, y2-5.8/2], 250+5.8);

    battenhorz([x1-5.8/2, y1+5.8/2], tp1, 0);
    battenhorz([x2+5.8/2, y2-5.8/2], tp2, 0);
    battenhorz([x2+5.8/2, y2-5.8/2], [x1-5.8/2, y2-5.8/2], 0);

}

room();
bed();
frame();