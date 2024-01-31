#!/bin/bash
rm -f Voelkerrecht_Glosar_*
tr '[\000-\037]' '\n' < Gesamtglossar.lst | tr -s '\n' | perl -e '
	use encoding "utf8", STDIN => "latin1";
	while (<>) {
		chomp;
		if (/\{\\rtf1\\ansi\\deff0/) {
			$title = $last_line;
			$title =~ s/^[-'\''1%*# &"()!]*//;
			s/^[^{]*\{/{/;
			$buffer = $_;
		} elsif ($buffer ne "") {
			$buffer .= " " . $_;
			if (/\}\}$/) {
				$buffer =~ s/\{\\fonttbl\{[^{}]*\}\{[^{}]*\}\}//;
				$buffer =~ s/\{\\colortbl[^{}]*\}//;
				$buffer =~ s/\{\\stylesheet\{[^{}]*\}\}//;
				$buffer =~ s/\{\\rtf1[^{}]*\{\\plain \\f1\\fs24 (.*\S) *\\par *\}\}/$1/;
				$buffer =~ s/\\'\''82/'\''/g;
				$buffer =~ s/\\'\''85/.../g;
				$buffer =~ s/\\'\''c4/Ä/g;
				$buffer =~ s/\\'\''d6/Ö/g;
				$buffer =~ s/\\'\''dc/Ü/g;
				$buffer =~ s/\\'\''df/ß/g;
				$buffer =~ s/\\'\''e0/a/g;
				$buffer =~ s/\\'\''e4/ä/g;
				$buffer =~ s/\\'\''e8/e/g;
				$buffer =~ s/\\'\''e9/e/g;
				$buffer =~ s/\\'\''f6/ö/g;
				$buffer =~ s/\\'\''fc/ü/g;
				printf "%s\t%s\n", $title, $buffer;
				$buffer = "";
			}
		}
		$last_line = $_;
	}
' | sort -R | split -l15 - Voelkerrecht_Glosar_
mmv -v Voelkerrecht_Glosar_\* Voelkerrecht_Glosar_#1.txt
