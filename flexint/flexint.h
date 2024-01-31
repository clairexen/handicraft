//
//  flexint -- a C++ flexible fixed point framework for hardware synthesis
//
//  Copyright (C) 2015  Clifford Wolf <clifford@clifford.at>
//
//  Permission to use, copy, modify, and/or distribute this software for any
//  purpose with or without fee is hereby granted, provided that the above
//  copyright notice and this permission notice appear in all copies.
//
//  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//

#ifndef FLEXINT_H
#define FLEXINT_H

#include <math.h>
#include <algorithm>
#include "BigInteger.hh"

template<int N, int E>
struct flexint
{
	BigInteger intval;
	long double floatval;


	// --------------------------------------
	// Contructors

	flexint() {
		intval = 0;
		floatval = 0;
	}

	template<int N1, int E1>
	flexint(const flexint<N1, E1> &other) {
		int ediff = E - E1;
		floatval = other.floatval;
		intval = other.intval;
		if (ediff > 0)
			intval = BigInteger(intval.getMagnitude() >> ediff, intval.getSign());
		else if (ediff < 0)
			intval = BigInteger(intval.getMagnitude() << -ediff, intval.getSign());
	}

	static inline flexint<N, E> from_bits(long m) {
		flexint<N, E> v;
		v.floatval = m * exp2(E);
		v.intval = m;
		return v;
	}

	static inline flexint<N, E> from_int(long m) {
		flexint<N, E> v;
		v.floatval = m;
		v.intval = m;
		if (E > 0)
			v.intval = BigInteger(v.intval.getMagnitude() >> E, v.intval.getSign());
		else if (E < 0)
			v.intval = BigInteger(v.intval.getMagnitude() << -E, v.intval.getSign());
		return v;
	}

	static inline flexint<N, E> from_float(long double m) {
		flexint<N, E> v;
		v.floatval = m;
		v.intval = long(m * exp2(-E));
		return v;
	}


	// --------------------------------------
	// Converters

	long to_bits() const {
		return this->intval.toLong();
	}

	long to_int() const {
		if (E > 0)
			return BigInteger(this->intval.getMagnitude() << E, this->intval.getSign()).toLong();
		else if (E < 0)
			return BigInteger(this->intval.getMagnitude() >> -E, this->intval.getSign()).toLong();
		else
			return this->intval.toLong();
	}

	long double to_double() const {
		return this->intval.toLong() * exp2(E);
	}


	// --------------------------------------
	// Arithmic Primitive Operators

	template<int N1, int E1>
	flexint<std::max(N, N1)+std::abs(E-E1)+1, std::min(E, E1)> operator+(const flexint<N1, E1> &other) const {
		flexint<std::max(N, N1)+std::abs(E-E1)+1, std::min(E, E1)> v, v1 = *this, v2 = other;
		v.floatval = this->floatval + other.floatval;
		v.intval = v1.intval + v2.intval;
		return v;
	}

	template<int N1, int E1>
	flexint<std::max(N, N1)+std::abs(E-E1)+1, std::min(E, E1)> operator-(const flexint<N1, E1> &other) const {
		flexint<std::max(N, N1)+std::abs(E-E1)+1, std::min(E, E1)> v, v1 = *this, v2 = other;
		v.floatval = this->floatval - other.floatval;
		v.intval = v1.intval - v2.intval;
		return v;
	}

	template<int N1, int E1>
	flexint<N+N1, E+E1> operator*(const flexint<N1, E1> &other) const {
		flexint<N+N1, E+E1> v;
		v.floatval = this->floatval * other.floatval;
		v.intval = this->intval * other.intval;
		return v;
	}


	// --------------------------------------
	// Arithmic Primitive Functions

	template<int N1, int E1, int N2, int E2>
	static inline flexint<N, E> add(const flexint<N1, E1> &a1, const flexint<N2, E2> &a2) {
		return a1 + a2;
	}

	template<int N1, int E1, int N2, int E2>
	static inline flexint<N, E> sub(const flexint<N1, E1> &a1, const flexint<N2, E2> &a2) {
		return a1 - a2;
	}

	template<int N1, int E1, int N2, int E2>
	static inline flexint<N, E> mul(const flexint<N1, E1> &a1, const flexint<N2, E2> &a2) {
		return a1 * a2;
	}
};

#endif
