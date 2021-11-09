(set-option :produce-models true)
(set-logic ALL)

; using fsm- prefix for generic FSM API
(declare-datatype fsm-input (
  ; using myfsm- prefix for decarations specific to the test FSM
  (fsm-make-input (myfsm-input-value Bool))
))

(declare-datatype fsm-output (
  (fsm-make-output (myfsm-output-value Bool))
))

; state machine internal types
(declare-datatype myfsm-state ((myfsm-state-A) (myfsm-state-B) (myfsm-state-C) (myfsm-state-D)))

(declare-datatype fsm-state (
  (fsm-make-state (myfsm-state-value myfsm-state))
))

(declare-datatype fsm-time (
  ; a time step can either be the initial state ...
  (fsm-time-init)
  ; ... or be created from another time step and an input value
  (fsm-time-next (fsm-time-prev fsm-time) (fsm-time-input fsm-input))
))

; state machine definition
(define-funs-rec
  (
    (fsm-get-depth ((t fsm-time)) Int)
    (fsm-no-loops ((t fsm-time)) Bool)
    (fsm-no-loops-worker ((t fsm-time) (s fsm-state)) Bool)

    (fsm-get-state ((t fsm-time)) fsm-state)
    (fsm-get-output ((s fsm-state) (input fsm-input)) fsm-output)
  )
  (
    ; fsm-get-depth
    (match t (
      ((fsm-time-init) 0)
      ((fsm-time-next prev input) (+ 1 (fsm-get-depth prev)))
    ))

    ; fsm-no-loops
    (fsm-no-loops-worker t (fsm-get-state t))

    ; fsm-no-loops-worker
    (match t (
      ((fsm-time-init) true)
      ((fsm-time-next prev input) (and (distinct (fsm-get-state prev) s) (fsm-no-loops-worker prev s)))
    ))

    ; fsm-get-state
    (match t (
      ((fsm-time-init)
        (fsm-make-state myfsm-state-A)
      )
      ((fsm-time-next prev input)
        (match (myfsm-state-value (fsm-get-state prev)) (
          ((myfsm-state-A) (fsm-make-state (ite (myfsm-input-value input) myfsm-state-B myfsm-state-A)))
          ((myfsm-state-B) (fsm-make-state (ite (myfsm-input-value input) myfsm-state-D myfsm-state-C)))
          ((myfsm-state-C) (fsm-make-state (ite (myfsm-input-value input) myfsm-state-C myfsm-state-D)))
          ((myfsm-state-D) (fsm-make-state (ite (myfsm-input-value input) myfsm-state-A myfsm-state-C)))
        ))
      )
    ))

    ; fsm-get-output
    (match (myfsm-state-value s) (
      ((myfsm-state-A) (fsm-make-output (ite (myfsm-input-value input)  true  true)))
      ((myfsm-state-B) (fsm-make-output (ite (myfsm-input-value input) false false)))
      ((myfsm-state-C) (fsm-make-output (ite (myfsm-input-value input)  true  true)))
      ((myfsm-state-D) (fsm-make-output (ite (myfsm-input-value input) false  true)))
    ))
  )
)

; ---- example query #1 ----
; create a trace of length 3 ending in state D

(push 1)
(echo "")
(echo "*** QUERY #1 ***")

(declare-const i0 fsm-input)
(declare-const i1 fsm-input)
(declare-const i2 fsm-input)

(define-fun t0 () fsm-time fsm-time-init)
(define-fun t1 () fsm-time (fsm-time-next t0 i0))
(define-fun t2 () fsm-time (fsm-time-next t1 i1))

(assert (= (myfsm-state-value (fsm-get-state t2)) myfsm-state-D))

(check-sat)
; (get-model)

(get-value (
  (myfsm-input-value i0)
  (myfsm-input-value i1)
  (myfsm-input-value i2)
))
(get-value (
  (myfsm-state-value (fsm-get-state t0))
  (myfsm-state-value (fsm-get-state t1))
  (myfsm-state-value (fsm-get-state t2))
))
(get-value (
  (myfsm-output-value (fsm-get-output (fsm-get-state t0) i0))
  (myfsm-output-value (fsm-get-output (fsm-get-state t1) i1))
  (myfsm-output-value (fsm-get-output (fsm-get-state t2) i2))
))

(pop 1)

; ---- example query #2 ----
; create a trace of length 3 ending in state D, incrementally

(push 1)
(echo "")
(echo "*** QUERY #2 ***")

(declare-const i0 fsm-input)
(define-fun t0 () fsm-time fsm-time-init)
(check-sat-assuming (
  (= (myfsm-state-value (fsm-get-state t0)) myfsm-state-D)
))

(declare-const i1 fsm-input)
(define-fun t1 () fsm-time (fsm-time-next t0 i0))
(check-sat-assuming (
  (= (myfsm-state-value (fsm-get-state t1)) myfsm-state-D)
))

(declare-const i2 fsm-input)
(define-fun t2 () fsm-time (fsm-time-next t1 i1))
(check-sat-assuming (
  (= (myfsm-state-value (fsm-get-state t2)) myfsm-state-D)
))

(get-value (
  (myfsm-input-value i0)
  (myfsm-input-value i1)
  (myfsm-input-value i2)
))
(get-value (
  (myfsm-state-value (fsm-get-state t0))
  (myfsm-state-value (fsm-get-state t1))
  (myfsm-state-value (fsm-get-state t2))
))
(get-value (
  (myfsm-output-value (fsm-get-output (fsm-get-state t0) i0))
  (myfsm-output-value (fsm-get-output (fsm-get-state t1) i1))
  (myfsm-output-value (fsm-get-output (fsm-get-state t2) i2))
))

(pop 1)

; ---- example query #3 ----
; create a trace of length 3 ending in state D, implicitly

(push 1)
(echo "")
(echo "*** QUERY #3 ***")

(declare-const t fsm-time)
(assert (= (fsm-get-depth t) 3))
;(assert (fsm-no-loops t))

(assert (myfsm-input-value (fsm-time-input (fsm-time-prev t))))
(assert (myfsm-input-value (fsm-time-input t)))

(check-sat)
; (get-model)

(get-value (
  (myfsm-input-value (fsm-time-input (fsm-time-prev t)))
  (myfsm-input-value (fsm-time-input t))
))
(get-value (
  (myfsm-state-value (fsm-get-state (fsm-time-prev (fsm-time-prev t))))
  (myfsm-state-value (fsm-get-state (fsm-time-prev t)))
  (myfsm-state-value (fsm-get-state t))
))

(assert (= (myfsm-state-value (fsm-get-state t)) myfsm-state-D))

(check-sat)
; (get-model)

(get-value (
  (myfsm-input-value (fsm-time-input (fsm-time-prev t)))
  (myfsm-input-value (fsm-time-input t))
))
(get-value (
  (myfsm-state-value (fsm-get-state (fsm-time-prev (fsm-time-prev t))))
  (myfsm-state-value (fsm-get-state (fsm-time-prev t)))
  (myfsm-state-value (fsm-get-state t))
))

(pop 1)
