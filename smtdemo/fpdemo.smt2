(set-logic ALL)
(define-sort FP () (_ FloatingPoint 8 24)) ; Float32
(define-fun  RM () (RoundingMode) roundNearestTiesToEven)

(declare-fun A () (FP))
(declare-fun B () (FP))
(declare-fun C () (FP))
(declare-fun D () (FP))

; --- Sequences starting with A ---
(define-fun AB   () (FP) (fp.add RM A   B))
(define-fun ABC  () (FP) (fp.add RM AB  C))
(define-fun ABCD () (FP) (fp.add RM ABC D))
(define-fun ABD  () (FP) (fp.add RM AB  D))
(define-fun ABDC () (FP) (fp.add RM ABD C))

(define-fun AC   () (FP) (fp.add RM A   C))
(define-fun ACB  () (FP) (fp.add RM AC  B))
(define-fun ACBD () (FP) (fp.add RM ACB D))
(define-fun ACD  () (FP) (fp.add RM AC  D))
(define-fun ACDB () (FP) (fp.add RM ACD B))

(define-fun AD   () (FP) (fp.add RM A   D))
(define-fun ADB  () (FP) (fp.add RM AD  B))
(define-fun ADBC () (FP) (fp.add RM ADB C))
(define-fun ADC  () (FP) (fp.add RM AD  C))
(define-fun ADCB () (FP) (fp.add RM ADC B))

; --- Sequences starting with B ---
(define-fun BA   () (FP) (fp.add RM B   A))
(define-fun BAC  () (FP) (fp.add RM BA  C))
(define-fun BACD () (FP) (fp.add RM BAC D))
(define-fun BAD  () (FP) (fp.add RM BA  D))
(define-fun BADC () (FP) (fp.add RM BAD C))

(define-fun BC   () (FP) (fp.add RM B   C))
(define-fun BCA  () (FP) (fp.add RM BC  A))
(define-fun BCAD () (FP) (fp.add RM BCA D))
(define-fun BCD  () (FP) (fp.add RM BC  D))
(define-fun BCDA () (FP) (fp.add RM BCD A))

(define-fun BD   () (FP) (fp.add RM B   D))
(define-fun BDA  () (FP) (fp.add RM BD  A))
(define-fun BDAC () (FP) (fp.add RM BDA C))
(define-fun BDC  () (FP) (fp.add RM BD  C))
(define-fun BDCA () (FP) (fp.add RM BDC A))

; --- Sequences starting with C ---
(define-fun CB   () (FP) (fp.add RM C   B))
(define-fun CBA  () (FP) (fp.add RM CB  A))
(define-fun CBAD () (FP) (fp.add RM CBA D))
(define-fun CBD  () (FP) (fp.add RM CB  D))
(define-fun CBDA () (FP) (fp.add RM CBD A))

(define-fun CA   () (FP) (fp.add RM C   A))
(define-fun CAB  () (FP) (fp.add RM CA  B))
(define-fun CABD () (FP) (fp.add RM CAB D))
(define-fun CAD  () (FP) (fp.add RM CA  D))
(define-fun CADB () (FP) (fp.add RM CAD B))

(define-fun CD   () (FP) (fp.add RM C   D))
(define-fun CDB  () (FP) (fp.add RM CD  B))
(define-fun CDBA () (FP) (fp.add RM CDB A))
(define-fun CDA  () (FP) (fp.add RM CD  A))
(define-fun CDAB () (FP) (fp.add RM CDA B))

; --- Sequences starting with D ---
(define-fun DB   () (FP) (fp.add RM D   B))
(define-fun DBC  () (FP) (fp.add RM DB  C))
(define-fun DBCA () (FP) (fp.add RM DBC A))
(define-fun DBA  () (FP) (fp.add RM DB  A))
(define-fun DBAC () (FP) (fp.add RM DBA C))

(define-fun DC   () (FP) (fp.add RM D   C))
(define-fun DCB  () (FP) (fp.add RM DC  B))
(define-fun DCBA () (FP) (fp.add RM DCB A))
(define-fun DCA  () (FP) (fp.add RM DC  A))
(define-fun DCAB () (FP) (fp.add RM DCA B))

(define-fun DA   () (FP) (fp.add RM D   A))
(define-fun DAB  () (FP) (fp.add RM DA  B))
(define-fun DABC () (FP) (fp.add RM DAB C))
(define-fun DAC  () (FP) (fp.add RM DA  C))
(define-fun DACB () (FP) (fp.add RM DAC B))

; --- X uses a tree-summation ---
(define-fun X    () (FP) (fp.add RM AB CD))

; --- X must be different from the other sums ---
(assert (distinct X ABCD))
(assert (distinct X ABDC))
(assert (distinct X ACBD))
(assert (distinct X ACDB))
(assert (distinct X ADBC))
(assert (distinct X ADCB))
(assert (distinct X BACD))
(assert (distinct X BADC))
(assert (distinct X BCAD))
(assert (distinct X BCDA))
(assert (distinct X BDAC))
(assert (distinct X BDCA))
(assert (distinct X CBAD))
(assert (distinct X CBDA))
(assert (distinct X CABD))
(assert (distinct X CADB))
(assert (distinct X CDBA))
(assert (distinct X CDAB))
(assert (distinct X DBCA))
(assert (distinct X DBAC))
(assert (distinct X DCBA))
(assert (distinct X DCAB))
(assert (distinct X DABC))
(assert (distinct X DACB))

(check-sat)
(get-value (A B C D X
ABCD ABDC ACBD ACDB ADBC ADCB
BACD BADC BCAD BCDA BDAC BDCA
CBAD CBDA CABD CADB CDBA CDAB
DBCA DBAC DCBA DCAB DABC DACB))
