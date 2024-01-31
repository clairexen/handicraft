package fib

var n0, n1 = 0, 1

func Fib() (n int) {
	n, n0, n1 = n0, n1, n0+n1;
	return;
}
