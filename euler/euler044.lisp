;  Problem 44 - Project Euler
;  http://projecteuler.net/index.php?section=problems&id=44
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

(defun solve_problem ()
	(let (pl pd pn p last-p sum diff)
		(flet
			(
				(next-p ()
					(incf pn)
					(let (( x (* pn (- (* 3 pn) 1) 1/2) ))
						(setf (gethash x pl) x)
					)
				)
				(is-p (x)
					(let*
						(
							(n (round (/ (+ 1 (expt (+ 1 (* 24 x)) 1/2)) 6)))
							(p (* n (- (* 3 n) 1) 1/2))
						)
						(= x p)
					)
				)
			)
			
			(setq pl (make-hash-table))
			(setq pd 0)
			(setq pn 0)

			(setq last-p 0)
			(loop
				(setq p (next-p))
				; (format t "~D: ~D~%" pn p)
				(maphash #'(lambda (key val)
						(declare (ignore val))
						(unless (= p key)
							(setq diff (abs (- p key)))
							(if (and (>= diff pd) (> pd 0))
								(remhash key pl)
								(when (gethash diff pl)
									(setq sum (+ p key))
									; (format t "  sum=~D(~D), diff=~D(~D)~%" sum (is-p sum) diff (is-p diff))
									(when (and (or (= pd 0) (< diff pd)) (is-p sum) (is-p diff))
										; (format t "D = ~D~%" diff)
										(setq pd diff)
									)
								)
							)
						)
					) pl
				)
				; (format t ".. ~D ~D~%" (- p last-p) pd)
				(when (and (> pd 0) (> (- p last-p) pd)) (return pd))
				(setq last-p p)
			)
		)
	)
)

(compile 'solve_problem)
(print (solve_problem))

