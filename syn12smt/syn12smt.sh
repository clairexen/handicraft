#!/bin/bash
#
# Copyright (C) 2020  Claire Xenia Wolf <claire@clairexen.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
#
# Inspired by
# https://www.youtube.com/watch?v=RgxUuSXMV50
#
#
# Why is this a shell script? Idk.. It kinda just happened. Sorry.
#

# number of measurements
N=3

# problem size
M=13

# solver command
solver="boolector --smt2"
#solver="z3"

# constrain to cleaned-up version
cleanup=true

# add symmetry breaking constraints
symbreak=false

if $cleanup; then
	if [ $N -ne 3 ] || [ $M -ne 13 ]; then
		cleanup=false
	else
		symbreak=true
	fi
fi

{
zero="#b"
one="#b"
msb="#b"
for ((i=0; i<M; i++)); do
	zero="${zero}0"
	one="${one}$((i == M-1))"
	msb="${msb}$((i == 0))"
done

cat<<-EOT
	(set-option :produce-models true)
	(set-logic QF_UFBV)

	(define-fun T () Bool true)
	(define-fun F () Bool false)

	(define-sort bv () (_ BitVec $M))
	(declare-fun f0a () bv)
EOT
for ((k=0; k<N; k++)); do
	echo -n "(declare-fun f${k}b ("
	for ((i=0; i<k; i++)); do
		[ $i -gt 0 ] && echo -n " "
		echo -n "Bool Bool"
	done
	echo ") bv)"
	echo -n "(declare-fun f$((k+1))a ("
	for ((i=0; i<=k; i++)); do
		[ $i -gt 0 ] && echo -n " "
		echo -n "Bool Bool"
	done
	echo ") bv)"
done
if $symbreak; then
	echo -n "(declare-fun f${N}b ("
	for ((i=0; i<N; i++)); do
		[ $i -gt 0 ] && echo -n " "
		echo -n "Bool Bool"
	done
	echo ") Bool)"
fi

echo
echo "(define-fun btcnt ((x bv)) bv (bvadd"
for ((c=0; c<M; c++)); do
	cat<<-EOT
		(ite (= ((_ extract $c $c) x) #b1) $one $zero)
	EOT
done
echo "))"

for ((c=0; c<M; c++)); do
for ((f=0; f<2; f++)); do
	secret="#b"
	for ((i=0; i<M; i++)); do
		secret="${secret}$((i == M-1-c))"
	done

	echo
	echo "(define-fun secret_$c$f () bv $secret)"
	echo "(define-fun sel0a_$c$f () bv (f0a))"

	for ((k=0; k<N; k++)); do
		echo -n "(define-fun sel${k}b_$c$f () bv (f${k}b"
		for ((i=0; i<k; i++)); do
			echo -n " res${i}a_$c$f res${i}b_$c$f"
		done
		echo "))"

		echo "(assert (= $zero (bvand sel${k}a_$c$f sel${k}b_$c$f)))"
		echo "(assert (= (btcnt sel${k}a_$c$f) (btcnt sel${k}b_$c$f)))"
		echo "(define-fun res${k}x_$c$f () Bool (distinct $zero (bvand sel${k}a_$c$f secret_$c$f)))"
		echo "(define-fun res${k}y_$c$f () Bool (distinct $zero (bvand sel${k}b_$c$f secret_$c$f)))"
		echo "(define-fun res${k}a_$c$f () Bool (ite (= #b0 #b$f) res${k}x_$c$f res${k}y_$c$f))"
		echo "(define-fun res${k}b_$c$f () Bool (ite (= #b0 #b$f) res${k}y_$c$f res${k}x_$c$f))"

		echo -n "(define-fun sel$((k+1))a_$c$f () bv (f$((k+1))a"
		for ((i=0; i<=k; i++)); do
			echo -n " res${i}a_$c$f res${i}b_$c$f"
		done
		echo "))"
	done
	if $symbreak; then
		echo -n "(define-fun sel${N}b_$c$f () Bool (f${N}b"
		for ((i=0; i<N; i++)); do
			echo -n " res${i}a_$c$f res${i}b_$c$f"
		done
		echo "))"
		if [ $c -ne $((M-1)) ]; then
			echo "(assert (= sel${N}b_$c$f (= #b1 #b$f)))"
		fi
	fi

	cat<<-EOT
		(define-fun result_$c$f () bv sel${N}a_$c$f)
		(assert (= result_$c$f secret_$c$f))
	EOT
done
done

if $symbreak; then
	outs=()
	inner=()
	collectinner=true
	nodegen() {
		if [ $1 -lt $N ]; then
			nodegen $(($1+1)) "$2 T F"
			nodegen $(($1+1)) "$2 F F"
			nodegen $(($1+1)) "$2 F T"
			if $collectinner; then
				inner[${#inner[@]}]="$(echo $2)"
			fi
		else
			outs[${#outs[@]}]="$(echo $2)"
			if ! grep -q T <( echo $2 ); then
				collectinner=false
			fi
		fi
	}
	mirror() {
		echo $* | sed -re 's/([TF]) ([TF])/\1\2/g; s/TF/F T/g; s/FT/T F/g; s/FF/F F/g;'
	}
	nodegen 0 ""
	echo
	echo "(define-fun validoutpair ((p bv) (q bv)) Bool (or (= p q) (= p (bvlshr q $one))))"
	for ((i=0; i<$(((${#outs[@]}-1)/2)); i++)); do
		echo "(assert (validoutpair (f${N}a ${outs[i]}) (f${N}a ${outs[i+1]})))"
		echo "(assert (= (f${N}a ${outs[i]}) (f${N}a ${outs[${#outs[@]}-i-1]})))"
	done
	echo "(assert (= (f${N}a ${outs[(${#outs[@]}-1)/2]}) $msb))"
	for node in "${inner[@]}"; do
		other="$(mirror $node)"
		if [ "$other" != "$node" ]; then
			i=$(($(echo $node | wc -w) / 2))
			echo "(assert (= (f${i}a $node) (f${i}a $other)))"
			echo "(assert (= (f${i}b $node) (f${i}b $other)))"
		fi
	done
fi

if $cleanup; then
	echo
	cat<<-EOT
		(assert (= f0a #b0000001010101))
		(assert (= f0b #b0000010101010))

		(assert (=
			(btcnt (f1a F T))
			(btcnt (f1a T F))
			#b0000000000011
		))

		(assert (=
			(btcnt (f1a F F))
			#b0000000000010
		))

		(assert (=
			(btcnt (f2a F F F F))
			(btcnt (f2a F F F T))
			(btcnt (f2a F F T F))
			(btcnt (f2a F T F F))
			(btcnt (f2a F T F T))
			(btcnt (f2a F T T F))
			(btcnt (f2a T F F F))
			(btcnt (f2a T F F T))
			(btcnt (f2a T F T F))
			#b0000000000001
		))

		(assert (= (f3a F F F F F F) #b1000000000000))
		(assert (= (f3a F F F F F T) (f3a F F F F T F) #b0100000000000))

		(assert (= (f3a F F F T T F) (f3a F F T F F T) #b0010000000000))
		(assert (= (f3a F F F T F F) (f3a F F T F F F) #b0001000000000))
		(assert (= (f3a F F F T F T) (f3a F F T F T F) #b0000100000000))

		(assert (= (f3a F T T F T F) (f3a T F F T F T) #b0000010000000))
		(assert (= (f3a F T T F F F) (f3a T F F T F F) #b0000001000000))
		(assert (= (f3a F T T F F T) (f3a T F F T T F) #b0000000100000))

		(assert (= (f3a F T F F T F) (f3a T F F F F T) #b0000000010000))
		(assert (= (f3a F T F F F F) (f3a T F F F F F) #b0000000001000))
		(assert (= (f3a F T F F F T) (f3a T F F F T F) #b0000000000100))

		(assert (= (f3a F T F T T F) (f3a T F T F F T) #b0000000000010))
		(assert (= (f3a F T F T F F) (f3a T F T F F F) #b0000000000001))
		(assert (= (f3a F T F T F T) (f3a T F T F T F) #b0000000000000))
	EOT
	no13_worker() {
		local level=$1 args="$2"
		if [ $level -lt $N ]; then
			echo "(assert (= (bvand (f${level}a$args) #b1000000000000) #b0000000000000))"
			echo "(assert (= (bvand (f${level}b$args) #b1000000000000) #b0000000000000))"
			no13_worker $((level+1)) "$args F F"
			no13_worker $((level+1)) "$args F T"
			no13_worker $((level+1)) "$args T F"
		fi
	}
	no13_worker 0 ""
fi

echo
cat<<-EOT
	(check-sat)
	(get-model)
	(get-value (
EOT
get_value_worker() {
	local level=$1 args="$2"
	echo "(f${level}a$args)"
	if [ $level -lt $N ]; then
		echo "(f${level}b$args)"
		get_value_worker $((level+1)) "$args F F"
		get_value_worker $((level+1)) "$args F T"
		get_value_worker $((level+1)) "$args T F"
	fi
}
get_value_worker 0 ""
echo "))"
} > syn12smt.smt2

$solver syn12smt.smt2 | tee syn12smt.out
python3 makedot.py < syn12smt.out > syn12smt.dot
dot -Tpdf syn12smt.dot > syn12smt.pdf
