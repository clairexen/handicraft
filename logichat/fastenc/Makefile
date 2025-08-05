fastenc: fastenc.cc
	clang++ -o fastenc -Wall fastenc.cc

fastenc.cc: fastenc.re pattern.re
	re2c -T -o fastenc.cc fastenc.re

pattern.re: pattern.py ../tokens.py ../config.py
	../.venv/bin/python3 pattern.py

clean:
	rm -f fastenc fastenc.cc pattern.re pattern.cc
