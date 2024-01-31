
// ************************** captured nuts lock **************************

cnl1 = 1.9;     // screw diameter
cnl2 = 3.9;     // nut diameter
cnl3 = 1.5;     // nut thickness
cnl4 = 11;      // max screw length
cnl5 = 8;       // min screw length
cnl6 = 4;       // material thickness
cnl7 = 0.5;	// shift away from thin borders

module cnlA(toinner = [0, 0]) {
        translate(toinner * cnl7) circle(cnl1/2, $fn = 12);
}

module cnlB() {
        render(convexity = 2) {
                translate([ -cnl6, -cnl1/2 ]) square([ cnl4, cnl1 ]);
                translate([ cnl5-cnl6-cnl3, -cnl2/2 ]) square([ cnl3, cnl2 ]);
        }
}

