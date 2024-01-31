
$fn = 16;

mode =
//	"VQ44";
//	"VQ100";
	"LQFP32";

module qfboboard(
	// part parameters
	pinout, partsize, skip,
	
	// qf side parameters
	pitch, padlen, padextra, padinc, padsize,
	
	// breakout side parameters
	rm, space, linesize, holes
) {
	// calculated pad parameters
	function get_padbar_len(n) = (n-1) * pitch;
	padbar_len = [ get_padbar_len(pinout[0]), get_padbar_len(pinout[1]),
		get_padbar_len(pinout[2]), get_padbar_len(pinout[3]) ];
	
	// calculated pin parameters
	function get_pinbar_len(n) = (ceil(n/2)-1) * rm;
	pinbar_len = [ get_pinbar_len(pinout[0]), get_pinbar_len(pinout[1]),
		get_pinbar_len(pinout[2]), get_pinbar_len(pinout[3]) ];
	
	module pin() {
		difference() {
			union() {
				circle((rm-space-linesize)/2);
				child(0);
			}
			circle(holes/2);
		}
	}
	
	module pinbar(n) {
		for (i = [ 0 : ceil(n/2)-1 ]) translate([ i*rm, 0 ]) {
			translate([ 0, -rm ]) pin() translate([ -linesize/2, 0 ]) square([ linesize, rm/2 ]);
			translate([ 0, -rm/2 ]) circle(linesize / 2);
		}
		for (i = [ 0 : floor(n/2)-1 ]) translate([ i*rm, 0 ]) {
			translate([ 0, -2*rm ]) {
				pin()	rotate(-45) translate([ -linesize/2, 0 ]) square([ linesize, sin(45)*rm ]);
				translate([ -linesize/2 + rm/2, rm/2 ]) square([ linesize, rm ]);
				translate([ rm/2, rm/2 ]) circle(linesize / 2);
			}
			translate([ rm/2, -rm/2 ]) circle(linesize / 2);
		}
	}
	
	module padbar(n) {
		for (i = [ 0 : n-1 ]) translate([ i*pitch , 0 ])
		assign(x = padinc*((n-1)/2-abs((n-1)/2-i))) {
			translate([ -padsize/2, -padextra-x ]) square([ padsize, padlen+padextra+x ]);
			translate([ 0, -padextra-x ]) circle(padsize/2);
		}
	}
	
	module side(s) {
		n = pinout[s];
		translate([ -padbar_len[s]/2, -partsize[s%2]/2 ]) padbar(n);
		translate([ -pinbar_len[s]/2, -skip[s%2]*rm ]) pinbar(n);
		for (i = [ 0 : n -1 ])
			assign(padpos = [ i*pitch - padbar_len[s]/2, -padextra-padinc*((n-1)/2-abs((n-1)/2-i))-partsize[s%2]/2 ])
			assign(pinpos = [ i*rm/2 - pinbar_len[s]/2, -skip[s%2]*rm - rm/2 ])
			assign(len = sqrt(pow(pinpos[0]-padpos[0], 2) + pow(pinpos[1]-padpos[1], 2)))
			{
				translate(pinpos) rotate(atan2(pinpos[1]-padpos[1], pinpos[0]-padpos[0]) + 90) {
					polygon([ [ -linesize/2, 0 ], [ +linesize/2, 0 ], [ +padsize/2, len ], [ -padsize/2, len ] ]);
				}
			}
	}
	
	rotate(180) side(0);
	rotate(90) side(1);
	rotate(0) side(2);
	rotate(270) side(3);
}

if (mode == "VQ44")
	qfboboard(
		// part parameters
		pinout = [ 11, 11, 11, 11 ],
		partsize = [ 12, 12 ],
		skip = [ 3, 3 ],
		
		// qf side parameters
		pitch = 0.8,
		padlen = 1,
		padextra = 0.2,
		padinc = 0.5,
		padsize = 0.3,
		
		// breakout side parameters
		rm = 2.54,
		space = 0.4,
		linesize = 0.3,
		holes = 0.5
	);

if (mode == "VQ100")
	qfboboard(
		// part parameters
		pinout = [ 25, 25, 25, 25 ],
		partsize = [ 16, 16 ],
		skip = [ 6, 6 ],
		
		// qf side parameters
		pitch = 0.5,
		padlen = 1,
		padextra = 0.2,
		padinc = 0.5,
		padsize = 0.17,
		
		// breakout side parameters
		rm = 2.54,
		space = 0.4,
		linesize = 0.3,
		holes = 0.5
	);

if (mode == "LQFP32")
	qfboboard(
		// part parameters
		pinout = [ 8, 8, 8, 8 ],
		partsize = [ 9, 9 ],
		skip = [ 2, 2 ],
		
		// qf side parameters
		pitch = 0.8,
		padlen = 1,
		padextra = 0.2,
		padinc = 0.2,
		padsize = 0.4,
		
		// breakout side parameters
		rm = 2.54,
		space = 0.4,
		linesize = 0.3,
		holes = 0.5
	);
