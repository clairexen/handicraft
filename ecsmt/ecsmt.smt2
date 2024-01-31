; solving a simple elliptic curve-related problem with SMT2 ... not.
; see https://twitter.com/oe1cxw/status/963000306295558144

(set-option :produce-models true)
(set-logic QF_NIA)
(declare-fun A () Int)
(declare-fun B () Int)
(declare-fun C () Int)

(define-fun AB () Int (+ A B))
(define-fun AC () Int (+ A C))
(define-fun BC () Int (+ B C))

(assert (> A 0))
(assert (> B 0))
(assert (> C 0))

(assert (=
  (+
    (* A AB AC)
    (* B AB BC)
    (* C AC BC)
  )
  (* 4 AB AC BC)
))

; asserting a solution helps :D
;(assert (= A   4373612677928697257861252602371390152816537558161613618621437993378423467772036))
;(assert (= B  36875131794129999827197811565225474825492979968971970996283137471637224634055579))
;(assert (= C 154476802108746166441951315019919837485664325669565431700026634898253202035277999))

(check-sat)
(get-model)
