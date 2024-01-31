# a demo tree-link description file

demo = {
	-lm main.o
	# ax and bx are both exporting a x() function,
	# a is using the ax version and b is using the bx version
	{ a.o (ax.o) }
	{ b.o (bx.o) }
}

demo.so = {
	# here a() in demo.so is calling the ax version of x()
	# but plain x() is calling the bx version of x()
	{ a.o (ax.o) } bx.o
}

