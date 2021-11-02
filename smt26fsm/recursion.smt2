(set-logic ALL)

; ---- data types and helper functions ----

(declare-datatypes () (
  (fsm-state-sort fsm-state-A fsm-state-B fsm-state-C fsm-state-D)
  (fsm-input-sort fsm-input-0 fsm-input-1)
  (fsm-output-sort fsm-output-0 fsm-output-1)
  (fsm-time-sort t0 t1 t2)
))

(declare-fun fsm-time-input (fsm-time-sort) fsm-input-sort)
(declare-fun fsm-init-state () fsm-state-sort)

(define-fun fsm-time-precursor ((t fsm-time-sort)) fsm-time-sort
  (match t ((t2 t1) (t1 t0) (t0 t0))))

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

(define-fun-rec fsm-time-state ((t fsm-time-sort)) (fsm-state-sort)
  (let ((pre (fsm-time-precursor t))) (ite (= t pre) fsm-init-state
    (fsm-next-fun (fsm-time-state pre) (fsm-time-input pre)))))

; ---- example query ----

(declare-datatype fsm-info-tripple (
  (fsm-info (fsm-info-s fsm-state-sort) (fsm-info-i fsm-input-sort) (fsm-info-o fsm-output-sort))
))

(define-fun fsm-time-info ((t fsm-time-sort)) fsm-info-tripple
  (let ((s (fsm-time-state t)) (i (fsm-time-input t))) (fsm-info s i (fsm-output-fun s i))))

; ---- example query ----

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
