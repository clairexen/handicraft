(set-info :smt-lib-version 2.6)
(set-option :smtlib2_compliant true)
(set-option :verbose 255)
(set-option :timeout 500)
(set-option :opt.timeout 500)
(set-option :dump_models true)
(set-option :produce-models true)
(set-logic ALL)


; Declarations for the demo FSM
; =============================

(declare-datatype myfsm-state (
  (myfsm-state-A) (myfsm-state-B) (myfsm-state-C) (myfsm-state-D)
))

(define-fun myfsm-state-function ((state myfsm-state) (input Bool)) myfsm-state
	(match state (
		(myfsm-state-A (ite input myfsm-state-B myfsm-state-A))
		(myfsm-state-B (ite input myfsm-state-D myfsm-state-C))
		(myfsm-state-C (ite input myfsm-state-C myfsm-state-D))
		(myfsm-state-D (ite input myfsm-state-A myfsm-state-C))
	))
)

(define-fun myfsm-output-function ((state myfsm-state) (input Bool)) Bool
	(match state (
		(myfsm-state-A (ite input  true  true))
		(myfsm-state-B (ite input false false))
		(myfsm-state-C (ite input  true  true))
		(myfsm-state-D (ite input false  true))
	))
)


; Declarations for the generic FSM model/API
; ==========================================

(declare-datatype fsm-input ((fsm-make-input
	; List of all input signals
	(fsm-input-sig-din Bool)
)))

(declare-datatype fsm-output ((fsm-make-output
	; List of all output signals
	(fsm-output-sig-dout Bool)
)))

(declare-datatype fsm-state ((fsm-make-state
	; List of all state registers
	(fsm-state-reg-state myfsm-state)
)))

(declare-datatype fsm-time (
	; a time step can either be the initial state ...
	(fsm-time-init)
	; ... or be created from another time step and an input value
	(fsm-time-next (fsm-time-prev fsm-time) (fsm-time-input fsm-input))
))

(define-funs-rec (
	(fsm-get-depth ((t fsm-time)) Int)
	(fsm-no-loops ((t fsm-time)) Bool)
	(fsm-no-loops-worker ((t fsm-time) (s fsm-state)) Bool)

	(fsm-get-state ((t fsm-time)) fsm-state)
	(fsm-get-output ((t fsm-time)) fsm-output)
) (
	; fsm-get-depth
	(match t (
		(fsm-time-init 0)
		((fsm-time-next prev input) (+ 1 (fsm-get-depth prev)))
	))

	; fsm-no-loops
	(fsm-no-loops-worker t (fsm-get-state t))

	; fsm-no-loops-worker
	(match t (
		(fsm-time-init true)
		((fsm-time-next prev input) (and (distinct (fsm-get-state prev) s) (fsm-no-loops-worker prev s)))
	))

	; fsm-get-state
	(match t (
		(fsm-time-init (fsm-make-state
			; List of all initial register values
			myfsm-state-A
		))
		((fsm-time-next prev input) (fsm-make-state
			; List of all next register values
			(myfsm-state-function (fsm-state-reg-state (fsm-get-state prev)) (fsm-input-sig-din input))
		))
	))

	; fsm-get-output
	(match t (
		(fsm-time-init (fsm-make-output
			; List of all output signals
			false
		))
		((fsm-time-next prev input) (fsm-make-output
			; List of all output signals
			(myfsm-output-function (fsm-state-reg-state (fsm-get-state prev)) (fsm-input-sig-din input))
		))
	))
))


; Example Queries
; ===============

(declare-datatype fsm-info-data (
	(fsm-make-init-info)
	(fsm-make-info
		(state myfsm-state)
		(din Bool)
		(dout Bool)
		(next_state myfsm-state)
	)
))

(define-fun fsm-info ((t fsm-time)) fsm-info-data (match t (
	(fsm-time-init fsm-make-init-info)
	((fsm-time-next prev input) (fsm-make-info
		(fsm-state-reg-state (fsm-get-state prev))
		(fsm-input-sig-din input)
		(fsm-output-sig-dout (fsm-get-output t))
		(fsm-state-reg-state (fsm-get-state t))
	))
)))


; ---- example query #1 ----
; create a trace of length 3 ending in state D, explicit, one-shot

(push 1)
(echo "")
(echo "*** QUERY #1 ***")

(declare-const i0 fsm-input)
(declare-const i1 fsm-input)
(declare-const i2 fsm-input)

(define-fun t0 () fsm-time fsm-time-init)
(define-fun t1 () fsm-time (fsm-time-next t0 i0))
(define-fun t2 () fsm-time (fsm-time-next t1 i1))

(assert (= (fsm-state-reg-state (fsm-get-state t2)) myfsm-state-D))

(check-sat)
; (get-model)

(get-value ((fsm-info t0)))
(get-value ((fsm-info t1)))
(get-value ((fsm-info t2)))

(pop 1)


; ---- example query #2 ----
; create a trace of length 3 ending in state D, explicit, incrementally

(push 1)
(echo "")
(echo "*** QUERY #2 ***")

(declare-const i0 fsm-input)
(define-fun t0 () fsm-time fsm-time-init)
(check-sat-assuming ((= (fsm-state-reg-state (fsm-get-state t0)) myfsm-state-D)))

(declare-const i1 fsm-input)
(define-fun t1 () fsm-time (fsm-time-next t0 i0))
(check-sat-assuming ((= (fsm-state-reg-state (fsm-get-state t1)) myfsm-state-D)))

(declare-const i2 fsm-input)
(define-fun t2 () fsm-time (fsm-time-next t1 i1))
(check-sat-assuming ((= (fsm-state-reg-state (fsm-get-state t2)) myfsm-state-D)))

(get-value ((fsm-info t0)))
(get-value ((fsm-info t1)))
(get-value ((fsm-info t2)))

(pop 1)


; ---- example query #3 ----
; create a trace of length 3 ending in state D, implicitly

(push 1)
(echo "")
(echo "*** QUERY #3 ***")

(declare-const t fsm-time)
(assert (= (fsm-get-depth t) 2))

; == uncommenting this line causes Z3 to segfault ==
;(assert (fsm-no-loops t))

(assert (fsm-input-sig-din (fsm-time-input (fsm-time-prev t))))
(assert (fsm-input-sig-din (fsm-time-input t)))

; == CVC5 can't solve this, but Z3 can ==
(check-sat)
; (get-model)

(get-value ((fsm-info (fsm-time-prev (fsm-time-prev t)))))
(get-value ((fsm-info (fsm-time-prev t))))
(get-value ((fsm-info t)))

(get-value ((fsm-state-reg-state (fsm-get-state t))))
(assert (= (fsm-state-reg-state (fsm-get-state t)) myfsm-state-D))

; == neither Z3 nor CVC5 can solve this ==
(check-sat)
; (get-model)

(get-value ((fsm-info (fsm-time-prev (fsm-time-prev t)))))
(get-value ((fsm-info (fsm-time-prev t))))
(get-value ((fsm-info t)))

(pop 1)
