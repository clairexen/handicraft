;  Problem 5 - Project Euler
;  http://projecteuler.net/index.php?section=problems&id=5
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

(defun factors (n)
	(let ( (x (floor (expt n 1/2))) )
		(loop
			(when (= x 1) (return (list n)))
			(when (= (mod n x) 0) (return (append (factors x) (factors (/ n x)))))
			(setq x (- x 1))
		)
	)
)
(compile 'factors)

(setq fl (make-hash-table))
(setq result 1)

(do
        ( (x 1 (+ x 1)) )
        ( (> x 20) x )

	(format t "prime factors of ~D:" x)
	(setq fc (make-hash-table))

	(dolist (f (factors x))
		(if (gethash f fc)
			(incf (gethash f fc))
			(setf (gethash f fc) 1)
		)
	)

	(with-hash-table-iterator (next-entry fc)
		(loop (multiple-value-bind (more key value) (next-entry)
			(unless more (return))
			(format t " ~Dx~D" value key)
			(if (gethash key fl)
				(when (> value (gethash key fl)) (setf (gethash key fl) value))
				(setf (gethash key fl) value)
			)
		))
	)
	(format t "~%")
)

(format t "common prime factors: ")
(with-hash-table-iterator (next-entry fl)
	(loop (multiple-value-bind (more key value) (next-entry)
		(unless more (return))
		(setq result (* result (expt key value)))
		(format t " ~Dx~D" value key)
	))
)
(format t "~%")

(format t "result: ~D~%" result)

