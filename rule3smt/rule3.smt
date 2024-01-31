; Context:
; https://twitter.com/johnregehr/status/1067582912340480000

(set-logic ALL)
(define-fun R () Int 3)

; Any non-negative P

(declare-fun P () Int)
(assert (>= P 0))

; Any K in 0..9 and Q = 10*P + K

(declare-fun K () Int)
(assert (>= K 0))
(assert (< K 10))

(define-fun Q () Int (+ (* P 10) K))

; reminders

(define-fun P_REM () Int (mod P R))
(define-fun Q_REM () Int (mod Q R))

; alternative way of getting to Q_REM

(define-fun Q_REM_ALT () Int (mod (+ P_REM K) R))

; search for counter-example

(assert (distinct Q_REM Q_REM_ALT))
(check-sat)
