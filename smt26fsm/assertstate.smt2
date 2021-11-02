(set-logic ALL)

; ---- data types and helper functions ----

(declare-datatype fsm-state-sort ((fsm-state-A) (fsm-state-B) (fsm-state-C) (fsm-state-D)))
(declare-datatype fsm-input-sort ((fsm-input-0) (fsm-input-1)))
(declare-datatype fsm-output-sort ((fsm-output-0) (fsm-output-1)))
(declare-datatype fsm-time-sort ((t0) (t1) (t2)))

(declare-fun fsm-time-input (fsm-time-sort) fsm-input-sort)
(declare-fun fsm-time-state (fsm-time-sort) fsm-state-sort)
(declare-fun fsm-init-state () fsm-state-sort)

; ---- example state machine ----

(define-fun fsm-next-fun
  ((state fsm-state-sort) (input fsm-input-sort)) fsm-state-sort
  (match state (
  	(fsm-state-A (match input ((fsm-input-0 fsm-state-A) (fsm-input-1 fsm-state-B))))
  	(fsm-state-B (match input ((fsm-input-0 fsm-state-C) (fsm-input-1 fsm-state-D))))
  	(fsm-state-C (match input ((fsm-input-0 fsm-state-D) (fsm-input-1 fsm-state-C))))
  	(fsm-state-D (match input ((fsm-input-0 fsm-state-C) (fsm-input-1 fsm-state-A))))
  ))
)

(define-fun fsm-output-fun
  ((state fsm-state-sort) (input fsm-input-sort)) fsm-output-sort
  (match state (
  	(fsm-state-A (match input ((fsm-input-0 fsm-output-1) (fsm-input-1 fsm-output-1))))
  	(fsm-state-B (match input ((fsm-input-0 fsm-output-0) (fsm-input-1 fsm-output-0))))
  	(fsm-state-C (match input ((fsm-input-0 fsm-output-1) (fsm-input-1 fsm-output-1))))
  	(fsm-state-D (match input ((fsm-input-0 fsm-output-1) (fsm-input-1 fsm-output-0))))
  ))
)

; ---- FSM evaluator ----

(define-fun t2t ((A fsm-time-sort) (B fsm-time-sort)) Bool
	(= (fsm-next-fun (fsm-time-state A) (fsm-time-input A)) (fsm-time-state B))
)

; ---- query helpers ----

(declare-datatype fsm-info-tripple (
  (fsm-info (fsm-info-s fsm-state-sort) (fsm-info-i fsm-input-sort) (fsm-info-o fsm-output-sort))
))

(define-fun fsm-time-info ((t fsm-time-sort)) fsm-info-tripple
  (let ((s (fsm-time-state t)) (i (fsm-time-input t))) (fsm-info s i (fsm-output-fun s i))))

; ---- example query #1 ----

(echo "")
(echo "*** QUERY #1 ***")

; create (t0 -> t1 -> t2) graph
(assert (t2t t0 t1))
(assert (t2t t1 t2))

; initial state is A at t0
(assert (= (fsm-time-state t0) fsm-state-A))

; final state is D at t2 (with o=0)
(assert (= (fsm-time-state t2) fsm-state-D))
(assert (= (fsm-output-fun (fsm-time-state t2) (fsm-time-input t2)) fsm-output-0))

; find path
(check-sat)

(get-value (
  (fsm-time-info t0)
  (fsm-time-info t1)
  (fsm-time-info t2)
))

; ---- example query #2 ----

(echo "")
(echo "*** QUERY #2 ***")

(push)
(assert (= (fsm-output-fun (fsm-time-state t0) (fsm-time-input t0)) fsm-output-0))
(check-sat)
(pop)

; ---- example query #3 ----

(echo "")
(echo "*** QUERY #3 ***")

(assert (= (fsm-output-fun (fsm-time-state t0) (fsm-time-input t0)) fsm-output-1))

(check-sat)

(get-value (
  (fsm-time-info t0)
  (fsm-time-info t1)
  (fsm-time-info t2)
))

(echo "")
