(set-option :produce-models true)
(set-logic QF_AUFBV)

%%

; <module>_s is the sort (data type) used to represent a state of our verilog module
; we declare variables for 8 states
(declare-fun state0 () counter_s)
(declare-fun state1 () counter_s)
(declare-fun state2 () counter_s)
(declare-fun state3 () counter_s)
(declare-fun state4 () counter_s)
(declare-fun state5 () counter_s)
(declare-fun state6 () counter_s)
(declare-fun state7 () counter_s)

; state0 must be an initial state (i.e. register values must match initial values)
(assert (counter_i state0))

; state<n> to state<n+1> must be a valif state transition
; i.e. the state variables must represent a sequence
(assert (counter_t state0 state1))
(assert (counter_t state1 state2))
(assert (counter_t state2 state3))
(assert (counter_t state3 state4))
(assert (counter_t state4 state5))
(assert (counter_t state5 state6))
(assert (counter_t state6 state7))

; we are looking for a counter-example, i.e. at least one state must violate Verilog assertions
(assert (not (and
   (counter_a state0)
   (counter_a state1)
   (counter_a state2)
   (counter_a state3)
   (counter_a state4)
   (counter_a state5)
   (counter_a state6)
   (counter_a state7)
)))

; see if we can find a counter-example for our Verilog assertions
; if UNSAT no counter example exists, i.e. Verilog assertions hold (within the bound)
(check-sat)

; extract model (i.e. the counter-example) from solver
(get-value (
   (counter_a state0)
   (counter_a state1)
   (counter_a state2)
   (counter_a state3)
   (counter_a state4)
   (counter_a state5)
   (counter_a state6)
   (counter_a state7)
))
(get-value (
   (|counter_n cnt| state0)
   (|counter_n cnt| state1)
   (|counter_n cnt| state2)
   (|counter_n cnt| state3)
   (|counter_n cnt| state4)
   (|counter_n cnt| state5)
   (|counter_n cnt| state6)
   (|counter_n cnt| state7)
))

