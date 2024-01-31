;  Problem 9 - Project Euler
;  http://projecteuler.net/index.php?section=problems&id=9
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

(do
	( (b 1 (+ b 1)) )
	( (> b 500) NIL )
	(do
		( (a 1 (+ a 1)) )
		( (> a b) NIL )
		(setq c (- 1000 (+ a b)))
		; (format t "?? ~D^2 + ~D^2 = ~D^2~%" a b c)
		(when (= (+ (* a a) (* b b)) (* c c)) (format t "~D^2 + ~D^2 = ~D^2  =>  ~D~%" a b c (* a b c)))
	)
)

