package fib

import "testing"

func TestFib(t *testing.T) {
	i := 0;
	tst := func(n int) {
		i++;
		if x := Fib(); x != n {
			t.Errorf("%d'th fibonacci number is %d, but Fib() returned %d!", i, n, x)
		}
	};
	tst(0);
	tst(1);
	tst(1);
	tst(2);
	tst(3);
	tst(5);
	tst(8);
	tst(13);
	tst(21);
	tst(34);
	tst(55);
	tst(89);
	tst(144);
}
