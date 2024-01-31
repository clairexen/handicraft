#!/usr/bin/perl -w

use strict;

my @data;
push @data, [] for (0..20);

while (<>) {
	chomp;
	s/^\s*//;
	my @d = split /\s+/;
	push @{$data[$_]}, $d[$_] for (0..20);
}

my $max_time = 0;

for my $i (1..20) {
	printf "function servo_$i(i) = lookup(i, [";
	for my $j (0..$#{$data[$i]}) {
		my $a = $data[0][$j];
		my $b = $data[$i][$j];
		$max_time = $a if $a > $max_time;
		printf "," if $j > 0;
		printf "[$a,$b]";
	}
	printf "]);\n";
}

print << "EOT";
function spider_config(i) = [
	[ servo_1(i*$max_time), servo_2(i*$max_time) ],
	[ servo_3(i*$max_time), servo_4(i*$max_time), servo_5(i*$max_time) ],
	[ servo_6(i*$max_time), servo_7(i*$max_time), servo_8(i*$max_time) ],
	[ servo_9(i*$max_time), servo_10(i*$max_time), servo_11(i*$max_time) ],
	[ servo_12(i*$max_time), servo_13(i*$max_time), servo_14(i*$max_time) ],
	[ servo_15(i*$max_time), servo_16(i*$max_time), servo_17(i*$max_time) ],
	[ servo_18(i*$max_time), servo_19(i*$max_time), servo_20(i*$max_time) ]
];
EOT

