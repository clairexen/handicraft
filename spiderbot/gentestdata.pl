#!/usr/bin/perl -w

use strict;

my @rl = ( -30, -60, -30,    0,  10 );
my @rh = ( +30, +60, +30, +100, 180 );

for my $i (0 .. 10) {
	printf "%3d", $i;
	for my $j (0 .. 19) {
		my $l = $rl[$j < 2 ? $j : 2 + ($j-2)%3];
		my $h = $rh[$j < 2 ? $j : 2 + ($j-2)%3];
		printf " %3d", $l + int(rand($h-$l+1));
	}
	printf "\n";
}

