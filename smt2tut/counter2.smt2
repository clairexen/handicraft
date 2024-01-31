(set-option :produce-models true)
(set-logic QF_AUFBV)

(declare-sort state 0)

; state registers
(declare-fun state_cnt (state) (_ BitVec 3))

; init state
(define-fun state_init ((the_state state)) Bool (and
    (= (state_cnt the_state) #b000)
))

; state transition
(define-fun state_trans ((old_state state) (new_state state)) Bool (and
    (= (bvadd (bvmul (state_cnt old_state) #b010) #b011) (state_cnt new_state))
))

; verilog asserts
(define-fun state_asserts ((the_state state)) Bool (and
    (bvult (state_cnt the_state) #b100)
))

(declare-fun state0 () state)
(declare-fun state1 () state)
(declare-fun state2 () state)
(declare-fun state3 () state)
(declare-fun state4 () state)
(declare-fun state5 () state)
(declare-fun state6 () state)
(declare-fun state7 () state)

(assert (state_init state0))
(assert (state_trans state0 state1))
(assert (state_trans state1 state2))
(assert (state_trans state2 state3))
(assert (state_trans state3 state4))
(assert (state_trans state4 state5))
(assert (state_trans state5 state6))
(assert (state_trans state6 state7))

(assert (not (and
   (state_asserts state0)
   (state_asserts state1)
   (state_asserts state2)
   (state_asserts state3)
   (state_asserts state4)
   (state_asserts state5)
   (state_asserts state6)
   (state_asserts state7)
)))

(check-sat)

(get-value (
   (state_asserts state0)
   (state_asserts state1)
   (state_asserts state2)
   (state_asserts state3)
   (state_asserts state4)
   (state_asserts state5)
   (state_asserts state6)
   (state_asserts state7)
))

