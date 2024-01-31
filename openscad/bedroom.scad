// unit size: 1 cm

room_width = 200;
room_length = 505;
room_height = 280;

door_width = 80;
door_offset = 115;
door_height = 198;

window_width = 200 - (40+56);
window_offset = 56;
window_bot = 100;
window_top = 30;

module room_floor()
{
    translate([0, 0, -10]) cube([room_width, room_length, 10]);
}

module room_walls()
{
    render(convexity = 3) difference()
    {
        translate([-10, -10, -10]) 
            cube([room_width+20, room_length+20, room_height+10]);
        
        translate([0, 0, -20]) 
            cube([room_width, room_length, room_height+30]);
        
        translate([-20, door_offset, -20]) 
            cube([30, door_width, door_height+20]);

        translate([window_offset, -20, window_bot]) 
            cube([window_width, 30, room_height-window_top-window_bot]);
    }
}

module room_walls_cut()
{
    module b1() {
        translate([-50, -50, -50])
            cube([room_width+100, room_length+100, 160]);
    }

    module b2() {
        difference() {
            translate([-50, -50, 110])
                cube([room_width+100, room_length+100, room_height-50]);
            b3();
        }
    }
    
    module b3() {
        translate([50, 50, 130])
            cube([room_width, room_length, room_height-50]);
    }

    color("#c764a2") render(convexity=2) intersection() {
        room_walls();
        b1();
    }

    color("#ffeeee") render(convexity=2) intersection() {
        room_walls();
        b2();
    }
    
    %render(convexity=2) intersection() {
        room_walls();
        b3();
    }    
}

bed_w = 200;
bed_l = 160;
bed_h = 40;

module bed()
{
    color([0.9, 0.9, 0.8])
        cube([bed_w, bed_l, bed_h]);
}

upper_w = 90;
upper_l = 200;
upper_h = 170;
upper_t = 40;

module curtain()
{
    for (i = [1:10])
        color([0.8, 0.8, 0.8, 0.4])
        translate([-(room_width-upper_w-2)*i*0.1, -5, 2])
        rotate([0, 0, 10])
        cube([(room_width-upper_w-4)*0.08, 2, upper_h+upper_t-2]);
}

module upper()
{
    color([0.8, 0.7, 0.6])
    difference()
    {
        cube([upper_w, upper_l, upper_h+upper_t]);

        translate([-10, 10, -10])
            cube([upper_w+20, upper_l-20, upper_h+10]);

        translate([10, 10, upper_h+10])
            cube([upper_w-20, upper_l-20, upper_t]);
    }
    curtain();
}

table_w = 60;
table_l = 150;
table_h = 100;

module table()
{
    color([0.8, 0.7, 0.6])
    render(convexity=3) difference()
    {
        union() {
            cube([table_w, table_l, table_h]);
            translate([table_w-10, table_l/2, table_h])
                rotate([0, 90, 0]) cylinder(10, d=table_l);
        }

        translate([-10, 10, -10])
            cube([table_w+20, table_l-20, table_h]);

        translate([10, -10, -10])
            cube([table_w-20, table_l+20, table_h]);
    }    
}

billy_w = 30;
billy_l = 100;
billy_h = 130;

module billy()
{
    color([0.8, 0.7, 0.6])
    render(convexity=3) difference()
    {
        cube([billy_w, billy_l, billy_h]);

        translate([10, 10, 10])
            cube([billy_w, billy_l-20, 30]);
        translate([10, 10, 50])
            cube([billy_w, billy_l-20, 30]);
        translate([10, 10, 90])
            cube([billy_w, billy_l-20, 30]);
    }    
}

flag_l = 150;
flag_h = 100;

module flag()
{
    color("#55CDFC") cube([1, flag_l, flag_h/5]);
    translate([0, 0, flag_h/5])
        color("#F7A8B8") cube([1, flag_l, flag_h/5]);    
    translate([0, 0, 2*flag_h/5])
        color("#FFFFFF") cube([1, flag_l, flag_h/5]);    
    translate([0, 0, 3*flag_h/5])
        color("#F7A8B8") cube([1, flag_l, flag_h/5]);    
    translate([0, 0, 4*flag_h/5])
        color("#55CDFC") cube([1, flag_l, flag_h/5]);    
}

room_floor();
room_walls_cut();

translate([-0.5, room_length-flag_l/2-upper_l/2, 150])
    flag();

translate([0, room_length-bed_l-20, 0])
    bed();

translate([room_width-upper_w, room_length-upper_l, 0])
    upper();

translate([room_width-table_w, 0, 0])
    table();

billy();
