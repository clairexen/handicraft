;  Problem 3 - Project Euler
;  http://projecteuler.net/index.php?section=problems&id=3
;
;  Copyright (C) 2009  Clifford Wolf <clifford@clifford.at>
;
;  This program is free software; you can redistribute it and/or modify
;  it under the terms of the GNU General Public License as published by
;  the Free Software Foundation; either version 2 of the License, or
;  (at your option) any later version.
;
;  This program is distributed in the hope that it will be useful,
;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;  GNU General Public License for more details.
;
;  You should have received a copy of the GNU General Public License
;  along with this program; if not, write to the Free Software
;  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

(defun mpf (n)
	(let ( (x (floor (expt n 1/2))) )
		(loop
			(when (= x 1) (return n))
			(when (= (mod n x) 0) (return (max (mpf x) (mpf (/ n x)))))
			(setq x (- x 1))
		)
	)
)

(compile 'mpf)

; do the calculation 100x to ease benchmarking
(do
	( (x 1 (+ x 1)) )
	( (> x 100) x )
	(print (mpf 600851475143))
)

