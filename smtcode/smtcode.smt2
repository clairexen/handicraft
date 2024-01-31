(declare-const a Int)
(declare-const b Int)
(declare-const c Int)

(assert (<= 0 a 9))
(assert (<= 0 b 9))
(assert (<= 0 c 9))

(define-fun code () Int (+ (* 100 a) (* 10 b) c))

; -------------------------------------------

(define-fun one-of-three ((x Bool) (y Bool) (z Bool)) Bool (or
	(and x (not y) (not z))
	(and (not x) y (not z))
	(and (not x) (not y) z)
))

(define-fun two-of-three ((x Bool) (y Bool) (z Bool)) Bool (or
	(and (not x) y z)
	(and x (not y) z)
	(and x y (not z))
))

; -------------------------------------------

(define-fun hint_0_0 () Bool (= a 6))
(define-fun hint_0_1 () Bool (= b 8))
(define-fun hint_0_2 () Bool (= c 2))

(assert (one-of-three hint_0_0 hint_0_1 hint_0_2))

; -------------------------------------------

(define-fun hint_1_0 () Bool (or (= b 6) (= c 6)))
(define-fun hint_1_1 () Bool (or (= a 1) (= c 1)))
(define-fun hint_1_2 () Bool (or (= a 4) (= b 4)))

(assert (one-of-three hint_1_0 hint_1_1 hint_1_2))

; -------------------------------------------

(define-fun hint_2_0 () Bool (or (= b 2) (= c 2)))
(define-fun hint_2_1 () Bool (or (= a 0) (= c 0)))
(define-fun hint_2_2 () Bool (or (= a 6) (= b 6)))

(assert (two-of-three hint_2_0 hint_2_1 hint_2_2))

; -------------------------------------------

(define-fun hint_3_0 () Bool (or (= a 7) (= b 7) (= c 7)))
(define-fun hint_3_1 () Bool (or (= a 3) (= b 3) (= c 3)))
(define-fun hint_3_2 () Bool (or (= a 8) (= b 8) (= c 8)))

(assert (not (or hint_3_0 hint_3_1 hint_3_2)))

; -------------------------------------------

(define-fun hint_4_0 () Bool (or (= b 7) (= c 7)))
(define-fun hint_4_1 () Bool (or (= a 8) (= c 8)))
(define-fun hint_4_2 () Bool (or (= a 0) (= b 0)))

(assert (one-of-three hint_4_0 hint_4_1 hint_4_2))

; -------------------------------------------

(check-sat)
(get-value (code))

; -------------------------------------------

(assert (distinct code 42))
(check-sat)
