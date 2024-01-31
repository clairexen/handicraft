(set-option :produce-models true)
(set-logic QF_AUFBV)

(declare-fun state0 () (_ BitVec 3))
(assert (= state0 #b000))

(declare-fun state1 () (_ BitVec 3))
(assert (= state1 (bvadd state0 #b001)))

(declare-fun state2 () (_ BitVec 3))
(assert (= state2 (bvadd state1 #b001)))

(declare-fun state3 () (_ BitVec 3))
(assert (= state3 (bvadd state2 #b001)))

(declare-fun state4 () (_ BitVec 3))
(assert (= state4 (bvadd state3 #b001)))

(assert (not (and
   (bvult state0 #b100)
   (bvult state1 #b100)
   (bvult state2 #b100)
   (bvult state3 #b100)
   (bvult state4 #b100)
)))

(check-sat)
