
package mbk
import "testing"
import "fmt"

func TestExecute1(t *testing.T) {
	a, r := Parser("(print a b (quote c))", 0);
	fmt.Printf("<a:%s> <r:%s> =>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	context.Set(NewAId("b"), NewAId("x"));
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

func TestExecute2(t *testing.T) {
	a, r := Parser("#(x y z)", 0);
	fmt.Printf("<a:%s> <r:%s> =>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

func TestExecute3(t *testing.T) {
	a, r := Parser(`
		(group
		      [def 'test #|local
		          (args 'a1 'a2)
		          (print x a1 y a2 z)
		          a2
		      ]
		      (print [test 1 2])
		      (print | test 3 4)
		      a1
		)
	`, 0);
	fmt.Printf("<a:%s> <r:%s> =>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

func TestExecute4(t *testing.T) {
	a, r := Parser(`
		(group
		      (def 'lst '(A B C D E F G H I J K))
		      (block L1|loop
		           (print (first lst))
		           (set 'lst (rest lst))
		           (when (is-nil lst) (return hallo-world L1))
		      )
		)
	`, 0);
	fmt.Printf("<a:%s> <r:%s> =>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

func TestExecute5(t *testing.T) {
	a, r := Parser(`
		[group
		  (print (+ 4 3))
		  (print (+ 0.5 0.25))
		  (print (+ 1/2 1/4))
		  (print (- 4 3))
		  (print (- 0.5 0.25))
		  (print (- 1/2 1/4))
		  (print (* 4 3))
		  (print (* 0.5 0.25))
		  (print (* 1/2 1/4))
		  (* (+ a (- 10 1/2)) (* (* 3 5.5) b))
		]
	`, 0);
	fmt.Printf("<a:%s> <r:%s>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

func TestExecute6(t *testing.T) {
	a, r := Parser(`
		[group
		  (def 'mpf #|local
		    (args 'n)
		    (def 'x |floor (** n 1/2))
		    (block 'L1|loop
		      (when (= x 1) |return n 'L1)
		      (when (= [% n x] 0) |return (max (mpf x) (mpf (/ n x))) 'L1)
		      (set 'x (- x 1))
		    )
		  )
		  (mpf 13195)
		  ; (mpf 600851475143)
		]
	`, 0);
	fmt.Printf("<a:%s> <r:%s>\n", a, r);
	context := NewADict(nil);
	special := NewADict(nil);
	RegisterAllBuiltins(context);
	v, e := Execute(a, context, special);
	if e == nil {
		fmt.Printf("=> %s\n", v);
	} else {
		fmt.Printf("=> Detected %s (%v)\n", e, v);
	}
}

